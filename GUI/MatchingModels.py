from typing import Callable, List, Tuple
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import csv

import torch
from torch import Tensor, nn, optim, tensor
from torchvision import transforms, models
import pandas as pd


class FeaturesExtractor(nn.Module):
    "FeaturesExtractor intake a tensor repersent a photo are output its feathture"
    def __init__(self, model_constructor, weights=None, frozen=True) -> None:
        """make a model

        Args:
            model_constructor (_type_): _description_
            wieghts (_type_): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        super(FeaturesExtractor, self).__init__()
        model = model_constructor(
            weights=weights
        )  # create model with pretrained weight or None

        model.eval()

        if hasattr(model, "classifier") and not hasattr(model, "fc"):
            model.classifier = Identity()
        elif not hasattr(model, "classifier") and hasattr(model, "fc"):
            model.fc = Identity()
        elif not hasattr(model, "classifier") and not hasattr(model, "fc"):
            raise NotImplementedError(
                f"{model.__class__.__name__} have no features layer nor fc layer"
            )
        else:
            raise NotImplementedError(
                f"ERROR, {model.__class__.__name__} have both????????"
            )

        if frozen:
            for param in model.parameters():
                param.requires_grad = False

        self.name = model.__class__.__name__

        self.featurizer = model

    def forward(self, inputs: Tensor) -> Tensor:
        """take inputs tensor and do forward() to featurize the inputs image tensor,
        as different model have diffent output shape after featurise, this function also adjust the shpae

        Args:
            inputs (torch.Tensor): (B, C, H, W) C ==3

        Raises:
            NotImplementedError: Unexcepted shape of output

        Returns:
            torch.Tensor: _description_
        """
        outputs = self.featurizer(inputs)
        if type(outputs) != Tensor:
            outputs = outputs.logits  # googlenet and inception use namedtuple as output
        if outputs.dim() == 4:
            outputs = torch.nn.functional.adaptive_avg_pool2d(
                outputs, output_size=(1, 1)
            )
            outputs = torch.squeeze(outputs, dim=tuple(range(2, outputs.dim())))
        elif outputs.dim() == 2:
            pass
        else:
            raise NotImplementedError(
                f"Unexcepted deminsion shape {outputs.shape} from {self.name}"
            )
        outputs = nn.functional.normalize(outputs, p=2, dim=1)
        return outputs


class Identity(nn.Module):
    """An Identity layer"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class PairwiseDotproducts(nn.Module):
    """return the sum of pair wise dot product of a matrix"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """return the sum of pairwise dot product of a matrix

        Args:
            input (torch.Tensor): _description_

        Raises:
            NotImplementedError: Only intends for 2D tensor

        Returns:
            torch.Tensor: sum of pairwise dot product of a matrix, 1D Tensor
        """
        if input.dim() != 2:
            raise NotImplementedError(f"Unexecpted dim of {input.dim()}, should be two")
        #calulation ver2.0 around 25x faster on CPU/GPU
        vector_sum = input.sum(dim=0)
        norm_sum=input.square().sum()
        numer_of_product = input.shape[0]*input.shape[0]-input.shape[0]
        return (vector_sum.dot(vector_sum)-norm_sum)/numer_of_product
        
        #old impletation
        # output = torch.mm(input, torch.transpose(input, dim0=0, dim1=1))
        # output = torch.triu(output, diagonal=1).sum()
        # return output

class Inferencer:
    "Handle all the calulation, only place to mange inter-CPU/GPU data movement"
    def __init__(self, model: FeaturesExtractor, query_path:Path, target_path:Path, transform:Callable = None) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if transform is not None:
            self.transform = transform
        else:
            self.transform = ToTensor_and_Resize(224)
        self.transform = ToTensor_and_Resize(224) 
        self.model = model
        self.query_files, self.query_tensors = self.load_from_folder(query_path)
        self.target_files, self.target_tensors = self.load_from_folder(target_path)

        self.model.to(self.device)
        self.query_tensors=self.query_tensors.to(self.device)
        self.target_tensors = self.target_tensors.to(self.device)
        
    def load_from_folder(self, data_path:Path) -> tuple[List[Path],Tensor]:
        files, ouptus = [], []
        for file in data_path.glob("*"):
            files.append(file)
            ouptus.append(self.transform(Image.open(file)))
        ouptus = torch.stack(ouptus)
        return files, ouptus
    
    def find_k_most_similar(self, k=5):
        self.model.eval()
        if k > len(self.target_files):
            raise TypeError(f"{k=} is larger then target size={len(self.target_files)}")
        query_tensors = self.model(self.query_tensors)
        target_tensors = self.model(self.target_tensors).T

        cos_similaritys, top_classes = torch.mm(query_tensors, target_tensors).topk(k)
        cos_similaritys = cos_similaritys.tolist()
        top_classes = top_classes.tolist()

        # I know it can all be done in one-line list comprehension, but for better readability, I kept the loop
        result = []
        for query_file, cos_similarity, top_classe in zip(self.query_files, cos_similaritys, top_classes):
            matching_result = tuple( (sorce, self.target_files[target_index]) for sorce, target_index in zip(cos_similarity, top_classe))
            result.append((query_file, matching_result))
        return result
    
    def train_model(self):
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            criterion = PairwiseDotproducts()

            iter_num = 1
            for _ in range(iter_num):
                self.model.train()
                match_sapce = self.model(self.target_tensors)
                loss = criterion(match_sapce)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            self.model.eval()

def load_labeled_data_csv(data_path:Path, transforms_needed=None):
    
    with open(data_path/"record.csv") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        label_of= dict((query, target) for query, target in spamreader )


    target_tensors, match_files = [], []
    filename_mapto_index = dict()

    for i, file in enumerate((data_path/"targets").glob("*")):
        match_files.append(file)
        target_tensors.append(transforms_needed(Image.open(file)))
        filename_mapto_index[file.parts[-1]] = i
    target_tensors = torch.stack(target_tensors)
    
    input_tensors, input_files, labels = [], [], []
    for file in (data_path/"queries").glob("*"):
        input_files.append(file)
        input_tensors.append(transforms_needed(Image.open(file)))
        labels.append(filename_mapto_index[label_of[file.parts[-1]]])
    input_tensors = torch.stack(input_tensors)
    labels = tensor(labels).view(-1,1)

    return (input_tensors, input_files, labels), (target_tensors, match_files)

def load_data(data_path, transforms_needed=None):
    """NOT IN USE
    load data from path
    assume path has this file strure:
    [root path]
        -queries
            -[label]
                -*.jpg
            -[label]...
        -targets
            -[label].jpg
            -...

    Args:
        (data_path: Path): path object
        (transforms_needed:torchvision.transforms): transforms the import files

    Returns:
        (inputs: tensor(B, C, H, W), input_files: list[Path], labels: 2d - tensor(int)), (match_targets: tensor(B, C, H, W), match_labels: list[int]): as shown
    """

    if transforms_needed is None:
        transforms_needed = ToTensor_and_Resize(224)

    inputs_path = data_path / "queries"
    inputs, labels, input_files = [], [], []
    for file in inputs_path.glob("*/*"):
        input_files.append(file)
        labels.append(int(file.parts[-2]))
        inputs.append(transforms_needed(Image.open(file)))
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels).view(-1, 1)

    match_target_path = data_path / "targets"
    match_targets, match_files, match_labels = [], [], []
    for file in match_target_path.glob("*"):
        match_files.append(file)
        match_labels.append(int(file.stem))
        match_targets.append(transforms_needed(Image.open(file)))
    match_targets = torch.stack(match_targets)
    match_labels = torch.tensor(match_labels).view(-1, 1)

    return (inputs, input_files, labels), (match_targets, match_files, match_labels)


def random_sample_form_imagepair_dataset(
    data_path: Path, validate_csv: Path, n: int, transforms_needed: Callable
) -> tuple:
    """load data from data set and csv file for laels
    assume data_path has this file strure:
    [root path]
        - *.jpg
        - *.jpg
    assume csv has this file strure:
    index | col1 | col2

    each row in (col1, col2) is a image pair, with file name as value

    Args:
        data_path (Path): _description_
        validate_csv (Path): _description_
        n (_type_): _description_
        transforms_needed (transforms): _description_

    Returns:
        tuple: as shown
    """

    datafram = pd.read_csv(validate_csv, usecols=[1, 2])
    samples = datafram.sample(n, axis=0)

    inputs, input_files, labels = [], [], []
    match_targets, match_files, match_labels = [], [], []
    for _, query, ref in samples.itertuples():
        input_files.append(data_path / (str(query) + ".jpg"))
        labels.append(ref)
        inputs.append(transforms_needed(Image.open(input_files[-1])))

        match_files.append(data_path / (str(ref) + ".jpg"))
        match_labels.append(ref)  # to match index of matching matrix to file so use ref
        match_targets.append(transforms_needed(Image.open(match_files[-1])))
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels).view(-1, 1)
    match_targets = torch.stack(match_targets)
    match_labels = torch.tensor(match_labels).view(-1, 1)

    return (inputs, input_files, labels), (match_targets, match_files, match_labels)


def error_rate(
    labels: Tensor, outputs: Tensor, match_sapce: Tensor, k=2
):
    """return topk_error_rate, (cos_similaritys, top_classes)

    Returns:
        tuple(float, tuple(Tensor, Tensor)): topk_error_rate, (cos_similaritys, top_classes)
    """
    match_sapce = torch.transpose(match_sapce, dim0=0, dim1=1)
    cos_similaritys, top_classes = torch.mm(outputs, match_sapce).topk(k)

    # match the predecition(index of match_sapce) to file name of match samples
    topk_error_rate = top_classes.clone()
    topk_error_rate = torch.any(topk_error_rate == labels, dim=1)  # the correct rate
    topk_error_rate = 1 - (topk_error_rate.sum() / topk_error_rate.shape[0]).item()

    return topk_error_rate, (cos_similaritys, top_classes)


def ToTensor_and_Resize(n):
    transforms_needed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((n,n), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms_needed


def model_trainer(
    model: FeaturesExtractor,
    inputs: Tensor,
    match_targets: Tensor,
    labels: Tensor,
    k_in_topk=5,
    iter_num=15,
    learn_rate=0.00005,
    show_progress=False,
):
    """refactor for better readbility

    Args:
        model (FeaturesExtractor): model to be trained
        inputs (Tensor): tensors for query photos
        match_targets (Tensor): tensors for  targets photos
        labels (Tensor): label
        match_labels (Tensor):
        k_in_topk (int, optional): k in topk function. Defaults to 5.
        iter_num (int, optional): how many cycle to train. Defaults to 15.
        learn_rate (float, optional): learn_rate. Defaults to 0.00005.
        show_progress (bool, optional): show_progress . Defaults to False.

    Returns:
        _type_: _description_
    """

    loss_tracking, error_rate_tracking = [], []
    with torch.no_grad():
        model.eval()
        topk_error_rate, _ = error_rate(
            labels, model(inputs), model(match_targets), k=k_in_topk
        )
        error_rate_tracking.append(topk_error_rate)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = PairwiseDotproducts()  # sum of pairwaise dot-product as loss function

    for iter_index in range(iter_num):
        model.train()
        match_sapce = model(match_targets)
        loss = criterion(match_sapce)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tracking.append(loss.item())

        with torch.no_grad():
            model.eval()
            topk_error_rate, _ = error_rate(
                labels, model(inputs), model(match_targets), k=k_in_topk
            )
            error_rate_tracking.append(topk_error_rate)

        if show_progress:
            print(f"iter {iter_index} done", end=" ")

    return loss_tracking, error_rate_tracking


models_list = dict(
    AlexNet=models.alexnet,
    ConvNeXt=models.convnext_base,
    EfficientNet=models.efficientnet_b0,
    GoogLeNet=models.googlenet,
)

transforms_default = ToTensor_and_Resize(224)
