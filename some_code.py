from ast import Tuple
import torch
from torch import Tensor, nn, optim
from torchvision import transforms, models

from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

def show_helper(title: str, input_files: list[Path], labels: Tensor, top_classes: Tensor, match_files: list[Path], max_shown_entry = 5):
    """helper for shwon result
    ------------------------------------------
    Args:
        title (str): the title of chart, usually the model name
        input_files (list[Path]): the querys photos
        labels (Tensor): the labels of querys photos, same length as input_files
        top_classes (Tensor): the topk predication of querys photos, same length as input_files
        match_files (list[Path]): the target photos, index match with top_classes
        max_shown_entry (int, optional): How many photo to shown. Defaults to 5.
    """
    for i, (input_file, label, top_class) in enumerate(zip(input_files, labels, top_classes)):
        if i == max_shown_entry: 
            break
        top_class = top_class.tolist()
        fig = plt.figure(figsize=(10,7))

        qax = fig.add_subplot(1, len(top_class)+2, 1)
        qax.axis("off")
        qax.set_title(title)
        plt.imshow(Image.open(input_file))

        qax = fig.add_subplot(1, len(top_class)+2, 2)
        qax.axis("off")
        qax.set_title("label")
        plt.imshow(Image.open(match_files[0].parent/(str(label.item())+".jpg")))

        for i, predecition in enumerate(top_class, 3):
            predecition = Image.open(match_files[predecition])
            newax = fig.add_subplot(1, len(top_class)+2, i)
            newax.set_title(f"top {i-2}")
            newax.axis("off")
            plt.imshow(predecition)

class FeaturesExtractor(nn.Module):
    def __init__(self, model_constructor, weights = None, frozen = True) -> None:
        """make a model

        Args:
            model_constructor (_type_): _description_
            wieghts (_type_): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        super(FeaturesExtractor, self).__init__()
        model = model_constructor(weights = weights) # create model with pretrained weight or None

        model.eval()
        
        if hasattr(model, "classifier") and not hasattr(model, "fc"):
            model.classifier = Identity()
        elif not hasattr(model, "classifier") and hasattr(model, "fc"):
            model.fc = Identity()
        elif not hasattr(model, "classifier") and not hasattr(model, "fc"):
            raise NotImplementedError(f"{model.__class__.__name__} have no features layer nor fc layer")
        else:
            raise NotImplementedError(f"ERROR, {model.__class__.__name__} have both????????")
        
        if frozen:
            for param in model.parameters():
                    param.requires_grad = False
        
        self.name = model.__class__.__name__

        self.featurizer = model


    def forward(self,inputs: Tensor) -> Tensor:
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
            outputs = outputs.logits # googlenet and inception use namedtuple as output
        if outputs.dim() == 4:
            outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, output_size=(1,1))
            outputs = torch.squeeze(outputs, dim = tuple(range(2, outputs.dim())))
        elif outputs.dim() == 2:
            pass
        else:
            raise NotImplementedError(f"Unexcepted deminsion shape {outputs.shape} from {self.name}")
        outputs = nn.functional.normalize(outputs, p=2, dim=1)
        return outputs

class Identity(nn.Module):
    """ An Identity layer"""
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
        output = torch.mm(input, torch.transpose(input, dim0=0, dim1=1))
        output = torch.triu(output, diagonal=1).sum()
        return output

def load_data(data_path, transforms_needed = None):
    """load data from path
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
        transforms_needed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]) 
    
    inputs_path = data_path/"queries"
    inputs, labels, input_files = [], [], []
    for file in inputs_path.glob("*/*"):
        input_files.append(file)
        labels.append(int(file.parts[-2]))
        inputs.append(transforms_needed(Image.open(file)))
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels).view(-1,1)

    match_target_path = data_path/"targets"
    match_targets, match_files, match_labels= [], [], []
    for file in match_target_path.glob("*"):
        match_files.append(file)
        match_labels.append(int(file.stem))
        match_targets.append(transforms_needed(Image.open(file)))
    match_targets = torch.stack(match_targets)
    match_labels = torch.tensor(match_labels).view(-1,1)

    return (inputs, input_files, labels), (match_targets, match_files, match_labels)

def random_sample_form_imagepair_dataset(data_path: Path, validate_csv: Path, n: int, transforms_needed: transforms) -> tuple:
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
    
    datafram = pd.read_csv(validate_csv, usecols=[1,2])
    samples = datafram.sample(n, axis=0)

    inputs, input_files, labels = [], [], []
    match_targets, match_files, match_labels = [], [] ,[]
    for _, query, ref in samples.itertuples():

        input_files.append(data_path/(str(query)+".jpg"))
        labels.append(ref)
        inputs.append(transforms_needed(Image.open(input_files[-1])))
        
        match_files.append(data_path/(str(ref)+".jpg"))
        match_labels.append(ref) # to match index of matching matrix to file so use ref
        match_targets.append(transforms_needed(Image.open(match_files[-1])))
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels).view(-1,1)
    match_targets = torch.stack(match_targets)
    match_labels = torch.tensor(match_labels).view(-1,1)

    return (inputs, input_files, labels), (match_targets, match_files, match_labels)

def error_rate(labels: Tensor, match_labels: Tensor, outputs: Tensor, match_sapce: Tensor, k=2):
    """return topk_error_rate, (cos_similaritys, top_classes)

    Returns:
        tuple(float, tuple(Tensor, Tensor)): topk_error_rate, (cos_similaritys, top_classes)
    """
    match_sapce = torch.transpose(match_sapce, dim0=0, dim1=1)
    cos_similaritys, top_classes = torch.mm(outputs, match_sapce).topk(k)
    
    # match the predecition(index of match_sapce) to file name of match samples
    topk_error_rate = top_classes.clone().detach().to("cpu")
    topk_error_rate.apply_(lambda x: match_labels[x]) #convert from index to file name
    topk_error_rate = torch.any(topk_error_rate == labels, dim = 1) #the correct rate
    topk_error_rate = 1 - (topk_error_rate.sum()/topk_error_rate.shape[0]).item()
    
    return topk_error_rate, (cos_similaritys, top_classes)

def ToTensor_and_Resize(n):
    transforms_needed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(n, antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms_needed

def model_trainer(model: FeaturesExtractor, 
                 inputs: Tensor, 
                 match_targets:Tensor, 
                 labels: Tensor,  
                 match_labels:Tensor,  
                 k_in_topk = 5,
                 iter_num = 15,
                 learn_rate = 0.00005,
                 show_progress = False):
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
        topk_error_rate, _ = error_rate(labels, match_labels, model(inputs), model(match_targets), k = k_in_topk)
        error_rate_tracking.append(topk_error_rate)

    optimizer = optim.Adam(model.parameters(), lr = learn_rate)
    criterion = PairwiseDotproducts() # sum of pairwaise dot-product as loss function

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
            topk_error_rate, _ = error_rate(labels, match_labels, model(inputs), model(match_targets), k = 5)
            error_rate_tracking.append(topk_error_rate)

        if show_progress:
            print(f"iter {iter_index} done", end=" ")
    
    return loss_tracking,error_rate_tracking