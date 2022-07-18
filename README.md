# ScalableViT
This is the code of paper "ScalableViT: Rethinking the Context-oriented Generalization of Vision Transformer".

It currently includes code and models for the following tasks:

✅ Image Classification

✅ Object Detection

✅ Semantic Segmentation

## Introduction
**ScalableViT** (**Scalable** **Vi**sion **T**ransformer) inculdes Scalable Self-Attention (SSA) and Interactive Window-based Self-Attention (IWSA) mechanisms.
SSA leverages two scaling factors to release dimensions of $query$, $key$, and $value$ matrices.
IWSA establishes interaction between non-overlapping regions by re-merging independent $value$ tokens and aggregating spatial information from adjacent windows.
By stacking the SSA and IWSA alternately, ScalableViT-S achieves $83.1 \%$ acc on ImageNet-1K.

![Architecture](/figures/framework.png)

## Main results

### Image Classification on ImageNet
| Model         | #Param.(M) | FLOPs(G) | top1-acc |
| ------------- | ---------- | -------- | -------- |
| ScalableViT-S | 32.4       | 4.2      | 83.1     |
| ScalableViT-B | 81.9       | 8.6      | 84.1     |
| ScalableViT-L | 104.9      | 14.7     | 84.4     |

### Object Detection on COCO
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

### Semantic Segmentation on ADE20K
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




## Citation
```latex
@article{ScalableViT,
  title={ScalableViT: Rethinking the context-oriented generalization of vision transformer},
  author={Yang, Rui and Ma, Hailong and Wu, Jie and Tang, Yansong and Xiao, Xuefeng and Zheng, Min and Li, Xiu},
  journal={arXiv preprint arXiv:2203.10790},
  year={2022}
}
```
