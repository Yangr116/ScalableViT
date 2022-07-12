# ScalableViT
This is the code of paper "ScalableViT: Rethinking the Context-oriented Generalization of Vision Transformer".

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

| Model         | #Param.(M) | FLOPs(G) | top1-acc |
| ------------- | ---------- | -------- | -------- |
| ScalableViT-S | 32.4       | 4.2      | 83.1     |
| ScalableViT-B | 81.9       | 8.6      | 84.1     |
| ScalableViT-L | 104.9      | 14.7     | 84.4     |

### Training
To train ScalableVit-S,B,L on ImageNet using 8 GPUs for 300 epochs, please run below code under the `classification` fold:
```shell
# small
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model scalable_vit_small  --batch-size 128 --data-path ${path_to_imagenet} --dist-eval --drop-path 0.2
# base
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model scalable_vit_base  --batch-size 128 --data-path ${path_to_imagenet} --dist-eval --drop-path 0.5
# large
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model scalable_vit_large  --batch-size 128 --data-path ${path_to_imagenet} --dist-eval --drop-path 0.5
```
where `${path_to_image}` is the ImageNet save path.
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
 <br/>

## Detection
We use both of RetinaNet and Mask R-CNN to evaluate our ScalableViT-S,B,L on COCO2017 dataset. Two detection frameworks are based on the mmdetection https://github.com/open-mmlab/mmdetection .
### Dependency
We use `mmdet==2.10.0` and `mmcv-full==1.2.7` to build the detection frameworks. Please install the relative dependency:
```shell
pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install mmdet==2.10.0
```
### Data preparation
Please download the COCO2017 dataset and put it into the folder `detection/data` .

### Detection Results on COCO2017
#### RetinaNet
| Backbone      | Pretrain    | Lr Schd | #Param.(M) | FLOPs(G) | bbox mAP |
| ------------- | ----------- | ------- | ---------- | -------- | -------- |
| ScalableViT-S | ImageNet-1K | 1x      | 36.4       | 238      | 45.2     |
| ScalableViT-S | ImageNet-1K | 3x      | 36.4       | 238      | 47.8     |
| ScalableViT-B | ImageNet-1K | 1x      | 85.6       | 330      | 45.8     |
| ScalableViT-B | ImageNet-1K | 3x      | 85.6       | 330      | 48.0     |
| ScalableViT-L | ImageNet-1K | 1x      | 112.6      | 457      | 46.8     |
#### Mask R-CNN
| Backbone      | Pretrain    | Lr Schd | #Param.(M) | FLOPs(G) | bbox mAP | mask mAP |
| ------------- | ----------- | ------- | ---------- | -------- | -------- | -------- |
| ScalableViT-S | ImageNet-1K | 1x      | 46.3       | 256      | 45.8     | 41.7     |
| ScalableViT-S | ImageNet-1K | 3x      | 46.3       | 256      | 48.7     | 43.6     |
| ScalableViT-B | ImageNet-1K | 1x      | 94.9       | 349      | 46.6     | 42.1     |
| ScalableViT-B | ImageNet-1K | 3x      | 94.9       | 349      | 48.9     | 43.6     |
| ScalableViT-L | ImageNet-1K | 1x      | 121.4      | 477      | 47.6     | 42.9     |
### Training
To train the RetinaNet and Mask R-CNN using ScalableViT-S,B,L as backbone on COCO2017, please run below code under the `detection` folder:
```shell
 bash dist_train.sh configs/${config_file} 8
```
where the `${config_file}` is the config file saved in the `detection/configs`, including 10 files.
### Evaluation
To evaluate the mAP of the RetinaNet and Mask R-CNN using ScalableViT-S,B,L as backbone on COCO2017, please run below code under the `detection` folder:
```shell
bash dist_test.sh configs/${config_file} ${checkpoint_file} 8 --eval mAP
```
where the `${config_file}` is the config file saved in the `detection/configs` and the `${checkpoint_file}` is the weight file saved in the `detection/work_dirs`.

<br/>

## Segmentation
We use both of Semantic FPN and UperNet to evaluate our ScalableViT-S,B,L on ADE20K dataset. Two segmentation frameworks are based on the mmsegmentation https://github.com/open-mmlab/mmsegmentation.

### Dependency
We use `mmseg==0.13.0` and `mmcv-full==1.3.1` to build the segmentation frameworks. Please install the relative dependency:
```shell
pip install mmcv-full==1.3.1 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install mmseg==0.13.0
```
### Data preparation
Please downlown the ADE20K dataset and put it into the folder `segmentation/data`.
### Segmentation Results on ADE20K
#### Semantic FPN
| Backbone      | Method       | Crop Size | Lr Schd | #Param.(M) | FLOPs(G) | mIoU |
| ------------- | ------------ | --------- | ------- | ---------- | -------- | ---- |
| ScalableViT-S | Semantic FPN | 512x512   | 80K     | 30.4       | 174      | 44.9 |
| ScalableViT-B | Semantic FPN | 512x512   | 80K     | 79.0       | 270      | 48.4 |
| ScalableViT-L | Semantic FPN | 512x512   | 80K     | 105.5      | 402      | 49.4 |
#### UperNet
| Backbone      | Method  | Crop Size | Lr Schd | #Param.(M) | FLOPs(G) | mIoU | mIoU (ms+flip) |
| ------------- | ------- | --------- | ------- | ---------- | -------- | ---- | ------------- |
| ScalableViT-S | UperNet | 512x512   | 160K    | 56.5       | 931      | 48.5 | 49.4          |
| ScalableViT-B | UperNet | 512x512   | 160K    | 107.0      | 1029     | 49.5 | 50.4          |
| ScalableViT-L | UperNet | 512x512   | 160K    | 135.5      | 1162     | 49.7 | 50.7          |
### Training
To train the Semantic FPN using ScalableViT-S,B,L as backbone on ADE20K, please run below code under the `segmentation` folder:
```shell
bash dist_train.sh configs/${config_file} 4
```
To train the UperNet using ScalableViT-S,B,L as backbone on ADE20K, please run below code under the `segmentation` folder:
```shell
bash dist_train.sh configs/${config_file} 8
```
where the `${config_file}` is the config file saved in the `segmentation/configs`, including 10 files.
### Evaluation
To evaluate the mIoU of the Semantic FPN and UperNet using ScalableViT-S,B,L as backbone on ADE20K, please run below code under the `segmentation` folder:
```shell
# single-gpu testing
python test.py ${config_file} ${checkpoint_file} --eval mIoU

# multi-gpu testing
tools/dist_test.sh ${config_file} ${checkpoint_file} ${gpu_num} --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh ${config_file} ${checkpoint_file} ${gpu_num} --aug-test --eval mIoU
```
where the `${config_file}` is the config file saved in the `segmentation/configs` and the `${checkpoint_file}` is the weight file saved in the `detection/work_dirs`.

# Citation
```
@article{ScalableVit,
  title={Scalablevit: Rethinking the context-oriented generalization of vision transformer},
  author={Yang, Rui and Ma, Hailong and Wu, Jie and Tang, Yansong and Xiao, Xuefeng and Zheng, Min and Li, Xiu},
  journal={arXiv preprint arXiv:2203.10790},
  year={2022}
}
```
