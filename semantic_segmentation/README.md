
## Segmentation
We use both of Semantic FPN and UperNet to evaluate our ScalableViT-S,B,L on ADE20K dataset. Two segmentation frameworks are based on the mmsegmentation https://github.com/open-mmlab/mmsegmentation.

### Dependency
We use `mmseg==0.13.0` and `mmcv-full==1.3.1` to build the segmentation frameworks. Please install the relative dependency:
```shell
pip install mmcv-full==1.3.1 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install mmseg==0.13.0
```
### Data preparation
Please downlown the ADE20K dataset and put it into the folder `data`.
```
ln -s ${your ADE20K} data/ADEChallengeData2016
```

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
where the `${config_file}` is the config file saved in the `semantic_segmentation/configs`, including 10 files.
### Evaluation
To evaluate the mIoU of the Semantic FPN and UperNet using ScalableViT-S,B,L as backbone on ADE20K, please run below code under the `semantic_segmentation` folder:
```shell
# single-gpu testing
python test.py ${config_file} ${checkpoint_file} --eval mIoU

# multi-gpu testing
tools/dist_test.sh ${config_file} ${checkpoint_file} ${gpu_num} --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh ${config_file} ${checkpoint_file} ${gpu_num} --aug-test --eval mIoU
```
where the `${config_file}` is the config file saved in the `semantic_segmentation/configs` and the `${checkpoint_file}` is the weight file saved in the `semantic_segmentation/work_dirs`.
