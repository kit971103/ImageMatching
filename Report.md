### Overview

This notebook present a method to improve the preforence of reverse image search model without labeled data and able to achieve 3.2% top-5 error rate.

### Task
Identify the a fashion product in photo.

There are two groups of photos, one group(targets) represents the different products, and another group(queries) are photos taken by employees or users. The task is to match the query photo with the corresponding photo in the target group.

![Alt text](Report_files/task.jpg)

### Method
Create a similarity learning model, and improve its accuracy for image matching **without labeled data**.

The challenge is to construct a similarity model with no labeled data.
There are some reported methods for labeled data [1-3]. Template matching[4] might not work here as well, as the query is potentially far from nearly duplicate of target photos[5].

In this notebook, I am attempting to apply transfer learning on a pre-trained image classifier for a features extractor to embed the image into a vector. Then using cosine similarity as a distance metric to measure the image similarity.\
(note: many potential variations exist, see discussion for details)

![Alt text](Report_files/workingflow.jpg)

However, as the original image classifier was not trained on fashion data, the out-of-the-box performance is not good, this notebook also attempts to improve its performance without expensive manual data labeling of similar and similar pair.

The idea is to orthogonalize the target photos, as the embedded vector is usually ~1000(i.e. efficientnet features output is 1280). As long as the number of input photos is less than the dimension of the features vector, it's guaranteed to make all inputs vector orthogonal.

![Alt text](Report_files/Orthogonalization.jpg)

### Code

The following implementation is based on Python and Pytorch package.

Features extractor is made by substituting the classifier layer in pre-trained models from torchvision.models(https://pytorch.org/vision/stable/models.html) with a Identity layer(forward call returns its input). The implementation of Identity layer can be found in some_code.py.


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
from pathlib import Path
my_Gdrive_root = "/content/drive/MyDrive/Colab Notebooks/PyTorch"
data_path = Path("/content/drive/MyDrive/Colab Notebooks/PyTorch/UNIQOExample")
import sys
sys.path.append(my_Gdrive_root)
import some_code as K
from importlib import reload
reload(K)

import torch, torchvision
from torch import Tensor, nn, optim
from torchvision import transforms, models

from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

# to fix the convextnet hash issue (ref: https://github.com/pytorch/vision/issues/7744)
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return torch.hub.load_state_dict_from_url(self.url, *args, **kwargs)
torchvision.models._api.WeightsEnum.get_state_dict = get_state_dict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

    cuda
    


```python
result, model_names = {"DEFAULT": [], None:[]}, []
transforms_default = K.ToTensor_and_Resize(224)
models_to_try = [
    (models.alexnet, transforms_default), #min_size 63
    (models.densenet121, transforms_default), #29
    (models.convnext_tiny, transforms_default), #32
    (models.efficientnet_b0, transforms_default), #1
    (models.googlenet, transforms_default), #224
    (models.inception_v3, K.ToTensor_and_Resize(299)), #299
    (models.maxvit_t, transforms_default), #224
    (models.mnasnet0_5, transforms_default), #1
    (models.mobilenet_v3_small, transforms_default), #1
] #tuple(model_constructor, transforms for inputs image)

for model_constructor, transforms_needed in models_to_try:
    for weight in ["DEFAULT", None]:

        (inputs, input_files, labels), (match_targets, match_files, match_labels) = K.load_data(data_path, transforms_needed)
        inputs, match_targets = inputs.to(device), match_targets.to(device)

        # make model by architecture specified and turn off the classfier in the orginal model
        model = K.FeaturesExtractor(model_constructor, weights = weight, frozen = True)
        model.to(device)

        # featurize input and match samples, then normalize to 1(by row aka each sample)
        outputs = model(inputs)
        match_sapce = model(match_targets)
        topk_error_rate, (cos_similaritys, top_classes) = K.error_rate(labels, match_labels, outputs, match_sapce, k = 5)

        result[weight].append(topk_error_rate)
    model_names.append(model.name)

print(pd.DataFrame(result, index=model_names))
```

    Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
    100%|██████████| 233M/233M [00:02<00:00, 119MB/s]
    Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /root/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
    100%|██████████| 30.8M/30.8M [00:00<00:00, 258MB/s]
    Downloading: "https://download.pytorch.org/models/convnext_tiny-983f1562.pth" to /root/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth
    100%|██████████| 109M/109M [00:03<00:00, 34.1MB/s]
    Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth
    100%|██████████| 20.5M/20.5M [00:00<00:00, 180MB/s]
    Downloading: "https://download.pytorch.org/models/googlenet-1378be20.pth" to /root/.cache/torch/hub/checkpoints/googlenet-1378be20.pth
    100%|██████████| 49.7M/49.7M [00:01<00:00, 29.0MB/s]
    /usr/local/lib/python3.10/dist-packages/torchvision/models/googlenet.py:47: FutureWarning: The default weight initialization of GoogleNet will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.
      warnings.warn(
    Downloading: "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth" to /root/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth
    100%|██████████| 104M/104M [00:00<00:00, 200MB/s] 
    /usr/local/lib/python3.10/dist-packages/torchvision/models/inception.py:43: FutureWarning: The default weight initialization of inception_v3 will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    Downloading: "https://download.pytorch.org/models/maxvit_t-bc5ab103.pth" to /root/.cache/torch/hub/checkpoints/maxvit_t-bc5ab103.pth
    100%|██████████| 119M/119M [00:00<00:00, 273MB/s]
    Downloading: "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth" to /root/.cache/torch/hub/checkpoints/mnasnet0.5_top1_67.823-3ffadce67e.pth
    100%|██████████| 8.59M/8.59M [00:00<00:00, 160MB/s]
    Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
    100%|██████████| 9.83M/9.83M [00:00<00:00, 164MB/s]
    

                   DEFAULT      None
    AlexNet       0.322581  0.451613
    DenseNet      0.193548  0.516129
    ConvNeXt      0.096774  0.645161
    EfficientNet  0.129032  0.451613
    GoogLeNet     0.290323  0.516129
    Inception3    0.129032  0.516129
    MaxVit        0.096774  0.419355
    MNASNet       0.193548  0.806452
    MobileNetV3   0.161290  0.451613
    

As shown although the models were trained on the ImageNet dataset, which contains mostly animals and daily objects, the FeaturesExtractor is still able to distall meaningful features for fashion product matching.
The models top-5 error rates is around 10%, which is improvable.
As the distance metric is the cosine similarity, if the target vectors are more evenly spaced, would probably result in better resolution.
Thus we can use the sum of pairwise dot-products of target tensors as a loss function to train the FeaturesExtractor. By minimizing the sum of pairwise dot-product, the vectors are more orthogonal thus more distance from each other.


```python
(inputs, input_files, labels), (match_targets, match_files, match_labels) = K.load_data(data_path, transforms_default)
inputs, match_targets = inputs.to(device), match_targets.to(device)
model = K.FeaturesExtractor(models.alexnet, weights="DEFAULT", frozen=False)
model.to(device)
print(f"{model.name}:")

loss_tracking, error_rate_tracking = [], []
with torch.no_grad():
    model.eval()
    topk_error_rate, _ = K.error_rate(labels, match_labels, model(inputs), model(match_targets), k = 5)
    error_rate_tracking.append(topk_error_rate)

optimizer = optim.Adam(model.parameters(), lr = 0.00005)
criterion = K.PairwiseDotproducts() # sum of pairwaise dot-product as loss function

for iter_index in range(15):
    model.train()
    match_sapce = model(match_targets)
    loss = criterion(match_sapce)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_tracking.append(loss.item())

    with torch.no_grad():
        model.eval()
        topk_error_rate, _ = K.error_rate(labels, match_labels, model(inputs), model(match_targets), k = 5)
        error_rate_tracking.append(topk_error_rate)
    print(f"iter {iter_index} done", end=" ")

print(f"\n{model.name} Done, inital topk error rate = {error_rate_tracking[0]*100: .1f}%, best = {min(error_rate_tracking)*100 :.1f}%")

error_rate_tracking = pd.Series(error_rate_tracking)
plt.close()
plt.figure()
error_rate_tracking.plot(title=model.name, xlabel="iter cycle", ylabel="top 5 error rate")
```

    AlexNet:
    iter 0 done iter 1 done iter 2 done iter 3 done iter 4 done iter 5 done iter 6 done iter 7 done iter 8 done iter 9 done iter 10 done iter 11 done iter 12 done iter 13 done iter 14 done 
    AlexNet Done, inital topk error rate =  32.3%, best = 25.8%
    




    <Axes: title={'center': 'AlexNet'}, xlabel='iter cycle', ylabel='top 5 error rate'>




    
![png](Report_files/Report_9_2.png)
    


With Alexnet as an example, by orthogonazing the matching matrix the error rate is decreased at first few iteration. Also, this behavior is not limited to Alexnet. The following code shows the result for 9 pre-trained models.


```python
loss_tracking_all, error_rate_tracking_all = pd.DataFrame(), pd.DataFrame()
for model_constructor, transforms_needed in models_to_try:

    (inputs, input_files, labels), (match_targets, match_files, match_labels) = K.load_data(data_path, transforms_needed)
    inputs, match_targets = inputs.to(device), match_targets.to(device)

    model = K.FeaturesExtractor(model_constructor, weights="DEFAULT", frozen=False)
    model.to(device)
    print(f"{model.name}:")

    loss_tracking, error_rate_tracking = K.model_trainer(model, inputs, match_targets, labels, match_labels,
                                                         iter_num = 15, k_in_topk = 5, learn_rate=0.00005) #refactor from above
    print(f"{model.name} Done, inital topk error rate = {error_rate_tracking[0]*100: .1f}%, best = {min(error_rate_tracking)*100 :.1f}%\n")
    loss_tracking_all[model.name] = loss_tracking
    error_rate_tracking_all[model.name] = error_rate_tracking

if device == "cuda":
    torch.cuda.empty_cache()
plt.close()
plt.figure()
error_rate_tracking_all.plot(subplots = True, layout = (3,3))
```

    AlexNet:
    AlexNet Done, inital topk error rate =  32.3%, best = 25.8%
    
    DenseNet:
    DenseNet Done, inital topk error rate =  19.4%, best = 19.4%
    
    ConvNeXt:
    ConvNeXt Done, inital topk error rate =  9.7%, best = 3.2%
    
    EfficientNet:
    EfficientNet Done, inital topk error rate =  12.9%, best = 6.5%
    
    GoogLeNet:
    GoogLeNet Done, inital topk error rate =  29.0%, best = 25.8%
    
    Inception3:
    Inception3 Done, inital topk error rate =  12.9%, best = 12.9%
    
    MaxVit:
    MaxVit Done, inital topk error rate =  9.7%, best = 9.7%
    
    MNASNet:
    MNASNet Done, inital topk error rate =  19.4%, best = 19.4%
    
    MobileNetV3:
    MobileNetV3 Done, inital topk error rate =  16.1%, best = 12.9%
    
    




    array([[<Axes: >, <Axes: >, <Axes: >],
           [<Axes: >, <Axes: >, <Axes: >],
           [<Axes: >, <Axes: >, <Axes: >]], dtype=object)




    <Figure size 640x480 with 0 Axes>



    
![png](Report_files/Report_11_3.png)
    


As shown above, the performance improvement is not only seen in AlexNet. DenseNet, ConvNeXt, EfficientNet, GoogLeNet also showed improvement in error rate after orthogonazation. With ConvNeXt preform the best after around 7 training cycles. However this method does not work for all models, this possibly results from model architecture. See the discussion section.


```python
(inputs, input_files, labels), (match_targets, match_files, match_labels) = K.load_data(data_path, transforms_default)
inputs, match_targets = inputs.to(device), match_targets.to(device)
model = K.FeaturesExtractor(models.convnext_tiny, weights="DEFAULT", frozen=False)
model.to(device)
print(f"{model.name}:")

optimizer = optim.Adam(model.parameters(), lr = 0.00005)
criterion = K.PairwiseDotproducts() # sum of pairwaise dot-product as loss function

for iter_index in range(12):
    model.train()
    match_sapce = model(match_targets)
    loss = criterion(match_sapce)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Training Done")
model.eval()
with torch.no_grad():
    topk_error_rate, (cos_similaritys, top_classes) = K.error_rate(labels, match_labels, model(inputs), model(match_targets), k = 5)
    print(f"{model.name} top 5 error rate = {topk_error_rate*100: .1f}%")
    K.show_helper(model.name, input_files, labels, top_classes, match_files, max_shown_entry=10)
```

    ConvNeXt:
    Training Done
    ConvNeXt top 5 error rate =  3.2%
    


    
![png](Report_files/Report_13_1.png)
    



    
![png](Report_files/Report_13_2.png)
    



    
![png](Report_files/Report_13_3.png)
    



    
![png](Report_files/Report_13_4.png)
    



    
![png](Report_files/Report_13_5.png)
    



    
![png](Report_files/Report_13_6.png)
    



    
![png](Report_files/Report_13_7.png)
    



    
![png](Report_files/Report_13_8.png)
    



    
![png](Report_files/Report_13_9.png)
    



    
![png](Report_files/Report_13_10.png)
    


## Results and Discussion


### Is the improvemwnt a result of random walk?


```python
loss_tracking_all, error_rate_tracking_all = pd.DataFrame(), pd.DataFrame()
for model_constructor, transforms_needed in models_to_try:

    (inputs, input_files, labels), (match_targets, match_files, match_labels) = K.load_data(data_path, transforms_needed)
    inputs, match_targets = inputs.to(device), match_targets.to(device)

    model = K.FeaturesExtractor(model_constructor, weights=None, frozen=False) # set to NO pretrain weight
    model.to(device)
    print(f"{model.name}:")

    loss_tracking, error_rate_tracking = K.model_trainer(model, inputs, match_targets, labels, match_labels,
                                                         iter_num = 50, k_in_topk = 5, learn_rate=0.00005) #refactor from above
    print(f"{model.name} Done, inital topk error rate = {error_rate_tracking[0]*100: .1f}%, best = {min(error_rate_tracking)*100 :.1f}%\n")
    loss_tracking_all[model.name] = loss_tracking
    error_rate_tracking_all[model.name] = error_rate_tracking

if device == "cuda":
    torch.cuda.empty_cache()
plt.close()
plt.figure()
error_rate_tracking_all.plot(subplots = True, layout = (3,3))
```

    AlexNet:
    AlexNet Done, inital topk error rate =  48.4%, best = 45.2%
    
    DenseNet:
    DenseNet Done, inital topk error rate =  48.4%, best = 41.9%
    
    ConvNeXt:
    ConvNeXt Done, inital topk error rate =  64.5%, best = 38.7%
    
    EfficientNet:
    EfficientNet Done, inital topk error rate =  51.6%, best = 51.6%
    
    

    /usr/local/lib/python3.10/dist-packages/torchvision/models/googlenet.py:47: FutureWarning: The default weight initialization of GoogleNet will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.
      warnings.warn(
    

    GoogLeNet:
    GoogLeNet Done, inital topk error rate =  64.5%, best = 51.6%
    
    

    /usr/local/lib/python3.10/dist-packages/torchvision/models/inception.py:43: FutureWarning: The default weight initialization of inception_v3 will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.
      warnings.warn(
    

    Inception3:
    Inception3 Done, inital topk error rate =  41.9%, best = 35.5%
    
    MaxVit:
    MaxVit Done, inital topk error rate =  41.9%, best = 41.9%
    
    MNASNet:
    MNASNet Done, inital topk error rate =  64.5%, best = 48.4%
    
    MobileNetV3:
    MobileNetV3 Done, inital topk error rate =  38.7%, best = 38.7%
    
    




    array([[<Axes: >, <Axes: >, <Axes: >],
           [<Axes: >, <Axes: >, <Axes: >],
           [<Axes: >, <Axes: >, <Axes: >]], dtype=object)




    <Figure size 640x480 with 0 Axes>



    
![png](Report_files/Report_16_7.png)
    


As shown above, For a randomly initialized model, this results in random-walk-like behavior.

Repeat this experiment 50 times, we can do some statics inferrnece on it.


```python
# this cell take VERY LONG time to run!! (~ 75min in colab with T4 GPU)
from itertools import product

drawdown_records = {"DEFAULT": pd.DataFrame(), None: pd.DataFrame()}

for (model_constructor, transforms_needed), weight in product(models_to_try, ["DEFAULT", None]):

    (inputs, input_files, labels), (match_targets, match_files, match_labels) = K.load_data(data_path, transforms_needed)
    inputs, match_targets = inputs.to(device), match_targets.to(device)

    record = []
    for iter_id in range(50):
        model = K.FeaturesExtractor(model_constructor, weights = weight, frozen=False)
        model.to(device)

        _, error_rate_tracking = K.model_trainer(model, inputs, match_targets, labels, match_labels,
                                                            iter_num = 15, k_in_topk = 5, learn_rate=0.00005) #refactor from above
        print(f"cycle {iter_id+1} for {model.name} done", end=" ")

        record.append(1 - min(error_rate_tracking)/error_rate_tracking[0])
    drawdown_records[weight][model.name] = record
    print(f"{model.name} Done")

if device == "cuda":
    torch.cuda.empty_cache()
```
```
progress markers........
```

```python
avg_improvment = pd.concat([drawdown_records[None].mean(), drawdown_records["DEFAULT"].mean()], axis = 1)
avg_improvment = avg_improvment.T.rename(index = {0: "random", 1: "pretrain"})
print(avg_improvment)
```

               AlexNet  DenseNet  ConvNeXt  EfficientNet  GoogLeNet  Inception3  \
    random    0.146723    0.0545  0.360035      0.009062   0.012673    0.194463   
    pretrain  0.200000    0.0000  0.660000      0.485000   0.160000    0.000000   
    
                MaxVit   MNASNet  MobileNetV3  
    random    0.177305  0.027966     0.032056  
    pretrain  0.000000  0.000000     0.200000  
    

The improvement (measured in 1 - best_error_rate/init_error_rate) of pre-trained model is a few times bigger than improvement of the random model. Especially for AlexNet, ConvNeXt both showed significant improvements.

Assuming the improvement of models are normally distributed, with null hypothesis of (improvement of randomly initialized models have and same mean as improvement of pre-trained models). The result of Welch's t-test for unequal variances is shown below:


```python
import scipy.stats as stats
for col in drawdown_records["DEFAULT"]:
    result = stats.ttest_ind(drawdown_records["DEFAULT"][col], drawdown_records[None][col], equal_var=False)
    print(f" the p-value for {col}: \t{result.pvalue: .3e}")
```

     the p-value for AlexNet: 	 1.205e-05
     the p-value for DenseNet: 	 1.056e-06
     the p-value for ConvNeXt: 	 8.272e-14
     the p-value for EfficientNet: 	 3.461e-56
     the p-value for GoogLeNet: 	 2.619e-20
     the p-value for Inception3: 	 5.078e-14
     the p-value for MaxVit: 	 8.522e-15
     the p-value for MNASNet: 	 4.222e-03
     the p-value for MobileNetV3: 	 1.621e-25
    

    /usr/local/lib/python3.10/dist-packages/scipy/stats/_axis_nan_policy.py:523: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
      res = hypotest_fun_out(*samples, **kwds)
    

The p-value is well below 0.05, so it's very unlikely a result of a random event.
The normal probability plots for the randomly initialized models are shown below:


```python
plt.figure(figsize=(10, 10))
for i, col in enumerate(drawdown_records[None], 1):
    plt.subplot(3,3,i)
    stats.probplot(stats.zscore(drawdown_records[None][col]), dist="norm", plot=plt)
    plt.title(col)
plt.tight_layout()
plt.show()
```


    
![png](Report_files/Report_23_0.png)
    


Although due to the small size of the data set, there are steps in data, the general shape still follows a normal distribution, with exceptions of EfficientNet, GoogLeNet, MNASNet, and MobileNetV3.

The underlying distribution of improvement measurement is very unlikely to be the true normal distribution, however, this should able to give a decent estimation of the validity of this method.

### Why does this method only work for some models?

When we take a closer look at the result, the behaviors of models in training can be divided into three classes.
1. Steady decrease at first. (AlexNet, ConvNeXt)
2. Steady increase first. (MaxVit, Inception3, DenseNet, EfficientNet, MNASNet)
3. No clear trend. (GoogLeNet, MobileNetV3)

For AlexNet[11] and ConvNeXt[12], are the only two models with pure convolution layers as building blocks.
DenseNet[13] contains densely connected convolutional modules.
GoogLeNet[14] and Inception3[15]'s core budling block is Inception module.
EfficientNet[16], MNASNet[17], MobileNetV3[18] all contain residual modules or inverted residual modules.
MaxVit[19] is transformer architecture.

This implies that this method may only work for pure convolutional neural networks. The mechanism behind this is yet to be further studied.

### Why increased training result in poor accuracy?

One possibility is that after some iter the featurizer starts to "forget" what it learned from the training dataset(ImageNet), instead trying to minimize the dot-products, which itself will not improve quality of feature extraction.

### This method also work for differnet k


```python
(inputs, input_files, labels), (match_targets, match_files, match_labels) = K.load_data(data_path, transforms_default)
inputs, match_targets = inputs.to(device), match_targets.to(device)

error_rate_tracking_diff_k = pd.DataFrame()

for k_in_topk in range(1,6):

    model = K.FeaturesExtractor(models.alexnet, weights="DEFAULT", frozen=False)
    model.to(device)
    print(f"{model.name}:")

    _, error_rate_tracking = K.model_trainer(model, inputs, match_targets, labels, match_labels,
                                                         iter_num = 15, k_in_topk = k_in_topk, learn_rate=0.00005) #refactor from above
    print(f"{model.name} Done, inital topk error rate = {error_rate_tracking[0]*100: .1f}%, best = {min(error_rate_tracking)*100 :.1f}%\n")
    error_rate_tracking_diff_k[f"k = {k_in_topk}"] = error_rate_tracking

if device == "cuda":
    torch.cuda.empty_cache()
plt.close()
plt.figure()
plt.tight_layout()
error_rate_tracking_diff_k.plot(subplots = True, layout = (5,1))
```

    AlexNet:
    AlexNet Done, inital topk error rate =  71.0%, best = 25.8%
    
    AlexNet:
    AlexNet Done, inital topk error rate =  58.1%, best = 25.8%
    
    AlexNet:
    AlexNet Done, inital topk error rate =  51.6%, best = 25.8%
    
    AlexNet:
    AlexNet Done, inital topk error rate =  41.9%, best = 25.8%
    
    AlexNet:
    AlexNet Done, inital topk error rate =  32.3%, best = 25.8%
    
    




    array([[<Axes: >],
           [<Axes: >],
           [<Axes: >],
           [<Axes: >],
           [<Axes: >]], dtype=object)




    <Figure size 640x480 with 0 Axes>



    
![png](Report_files/Report_28_3.png)
    


### Future work on
1. if labeled data is available, try SIAMESE CNN
2. Use other kinds of metrics to determin the similarity, i.e. L2-norm, L1-norm for non-unity vectors
3. Other kinds of feather extraction methods, such as SIFT, SURF, BRIEF, BRISK, and FAST.

## Method and Impleatation

### Data

Due to the lack of such pairing datasets in the fashion domain. Most available fashion datasets[8-10] are image labeled with descriptions or classes, rather than a pairing datasets. The available image pair datasets are designed for facical reconigtion[6], and semantic similarity[7].

The data used for this notebook is collected by myself on an online store of a big fashion retailer. Thus the dataset is relatively small(17 target photos and 30 querys photos), which might diminish the viability of this method.

Thus one further improvement of this work is to collect more data, possibly by web crawler. Also wider range of data, such as including other stores and other sources, such as photos in the comment section on e-commerce. So that bias in the dataset is eliminated.

However, due to the lack of suitable datasets, skills for making web crawler, and resources to collect more data at this moment, this potentially biased dataset is used for preliminary results.

### Code

All code is written in Python with Pytorch package. All code and images used are available on Github(https://github.com/kit971103/ImageMatching).

## Reference
1-3 are works done on Similarity model with labeled data\
4-5 are Template Matching


1. Chatterjee, Moitreya; Luo, Yunan. Course material from University of Illinois, "Similarity Learning with (or without) Convolutional Neural Network" (PDF). https://slazebni.cs.illinois.edu/spring17/lec09_similarity.pdf
2. Learning a Distance Metric from Relative Comparisons, https://proceedings.neurips.cc/paper_files/paper/2003/file/d3b1fb02964aa64e257f9f26a31f72cf-Paper.pdf; (This paper is about metric learning for similat/dissimilar image pair)
3. Large Scale Online Learning of Image Similarity Through Ranking, https://www.jmlr.org/papers/volume11/chechik10a/chechik10a.pdf; (Use of triplet loss for image similarity metric learning.)
4. OpenCV Template Matching, https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
5. IMAGE MATCH GITHUB, https://github.com/rhsimplex/image-match
6. Labeled Faces in the Wild, https://vis-www.cs.umass.edu/lfw/index.html
7. Images Alike, semantic similar image pair, https://www.kaggle.com/datasets/alincijov/images-alike
8. Fashion Product Images Dataset, https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
9. Adidas Fashion Retail Products Dataset, https://www.kaggle.com/datasets/thedevastator/adidas-fashion-retail-products-dataset-9300-prod
10. Clothing dataset ,https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full
11. "One weird trick for parallelizing convolutional neural networks", https://arxiv.org/pdf/1404.5997.pdf
12. "A ConvNet for the 2020s",https://arxiv.org/pdf/2201.03545.pdf
13. "Densely Connected Convolutional Networks", https://arxiv.org/pdf/1608.06993.pdf
14. "Going deeper with convolutions", https://arxiv.org/pdf/1409.4842.pdf
15. "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/pdf/1512.00567.pdf
16. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", https://arxiv.org/pdf/1905.11946.pdf
17. "MnasNet: Platform-Aware Neural Architecture Search for Mobile", https://arxiv.org/pdf/1807.11626.pdf
18. "MobileNetV2: Inverted Residuals and Linear Bottlenecks", https://arxiv.org/pdf/1801.04381.pdf
19. "MaxViT: Multi-Axis Vision Transformer", https://arxiv.org/pdf/2204.01697.pdf

### Real Motivation
On a road trip with my friends, one of my friends who owns a fashion business needed to work in a hotel room, while watching he work, I found that most of his work was pretty mechanical. There are two major tasks:
1. Product matching, match the product photo taken by the employee to a sample photo from the supplier for the internal inventory system.
2. Table recognition, manual input sale data from photo containing a table of daily sale record
That why i startd to learning ML and pytorch.
