## Classification
### Dependency
Please install PyTorch 1.7.0+ , torchvision 0.8.1+ and timm==0.3.2:

```shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
```
or build the environments according to the `requirements.txt`
```
pip install requirements.txt
```

### Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision `datasets.ImageFolder`, and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### Classification Results on ImageNet-1K
The results on ImageNet-1K as shown below:

| Model         | #Param.(M) | FLOPs(G) | top1-acc |  weights |  logs|
| ------------- | ---------- | -------- | -------- | -------- | -----|
| ScalableViT-S | 32.4       | 4.2      | 83.1     |  [google](https://drive.google.com/file/d/1gCWsATBJmW3xwJaoCcxGO1Wbg51cPtzk/view?usp=sharing)  |  [logs](https://drive.google.com/file/d/1QcvWak1fKplbxF7FuwugUaBaCLOxMNVb/view?usp=sharing) |
| ScalableViT-B | 81.9       | 8.6      | 84.1     |  [google](https://drive.google.com/file/d/16MVmK4BGQpRH3Wz2VCOVzhTh2PFUFozH/view?usp=sharing)  |   [logs](https://drive.google.com/file/d/1MBInNLsxJhJKkZUatDwEh6o32exmM0Sq/view?usp=share_link)   |
| ScalableViT-L | 104.9      | 14.7     | 84.4     |  [google](https://drive.google.com/file/d/17LLK5PSssMpepxeAt6-uKSd4aaV99u5X/view?usp=sharing)  |   [logs](https://drive.google.com/file/d/1lDYQPm2FAGvzYcneJaR_x-UZ6kd3Tho_/view?usp=share_link)   |

### Training
To train ScalableVit-S,B,L on ImageNet using 8 GPUs for 300 epochs, please run below code under the `image_classification` fold:
```shell
# small
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model scalable_vit_small  --batch-size 128 --data-path ${path_to_imagenet} --dist-eval --drop-path 0.2
# base
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model scalable_vit_base  --batch-size 128 --data-path ${path_to_imagenet} --dist-eval --drop-path 0.5
# large
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model scalable_vit_large  --batch-size 128 --data-path ${path_to_imagenet} --dist-eval --drop-path 0.5
```
where `${path_to_image}` is the path of your ImageNet.
### Evaluation
To evaluate the performance of ScalableViT-S,B,L on ImageNet using one GPU, please run:
```shell
# small
python3 main.py --eval --resume ./scalable_vit_small/checkpoint_best.pth  --model scalable_vit_small --data-path path_to_imagenet
# base
python3 main.py --eval --resume ./scalable_vit_base/checkpoint_best.pth  --model scalable_vit_base --data-path path_to_imagenet
# large
python3 main.py --eval --resume ./scalable_vit_large/checkpoint_best.pth  --model scalable_vit_large --data-path path_to_imagenet
```
### Throughtput
```shell
CUDA_VISIBLE_DEVICES=1 python3 main.py --model ${model_name} --data-path ${path_to_image} --batch-size 64 --throughput
```
 <br/>