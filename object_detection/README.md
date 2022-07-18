
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

#### Mask R-CNN
| Backbone      | Pretrain    | Lr Schd | #Param.(M) | FLOPs(G) | bbox mAP | mask mAP |
| ------------- | ----------- | ------- | ---------- | -------- | -------- | -------- |
| ScalableViT-S | ImageNet-1K | 1x      | 46.3       | 256      | 45.8     | 41.7     |
| ScalableViT-S | ImageNet-1K | 3x      | 46.3       | 256      | 48.7     | 43.6     |
| ScalableViT-B | ImageNet-1K | 1x      | 94.9       | 349      | 46.6     | 42.1     |
| ScalableViT-B | ImageNet-1K | 3x      | 94.9       | 349      | 48.9     | 43.6     |

### Training
To train the RetinaNet and Mask R-CNN using ScalableViT-S,B,L as backbone on COCO2017, please run below code under the `detection` folder:
```shell
 bash dist_train.sh configs/${config_file} 8
```
where the `${config_file}` is the config file saved in the `configs`, including 10 files.
### Evaluation
To evaluate the mAP of the RetinaNet and Mask R-CNN using ScalableViT-S,B,L as backbone on COCO2017, please run below code under the `detection` folder:
```shell
bash dist_test.sh configs/${config_file} ${checkpoint_file} 8 --eval mAP
```
where the `${config_file}` is the config file saved in the `configs` and the `${checkpoint_file}` is the weight file saved in the `./work_dirs`.

<br/>
