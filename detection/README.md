# FocalNets for Object Detection

## Installation and Data Preparation

We use [mmdetection](https://github.com/open-mmlab/mmdetection) and follow [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) to set up our pipelines. 

## Evaluation

To evaluate a pre-trained FocalNets on COCO, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 tools/test.py \
<config-file> <ckpt-path> --cfg-options data.samples_per_gpu=<samples_per_gpu> --luancher pytorch
```

For example, to evaluate the Mask R-CNN 1x model with FocalNet-B (LRF) on 8 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 tools/test.py \
configs/focalnet/mask_rcnn_focalnet_base_patch4_mstrain_480-800_adamw_1x_coco.py focalnet_base_lrf_maskrcnn_1x.pth \
--cfg-options data.samples_per_gpu=1 model.backbone.focal_levels='[3,3,3,3]'
```

## Training

To train a detection model with pretrained FocalNet, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  tools/train.py \
<config-file> \
--cfg-options \
model.pretrained=<pretrained/model/path> \
data.samples_per_gpu=<samples_per_gpu> \
--launcher pytorch
```

For example, we train mask r-cnn with following commands:

<details>

<summary>
Mask R-CNN with FocalNet-T
</summary>

FocalNet-T (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/mask_rcnn_focalnet_tiny_patch4_mstrain_480-800_adamw_1x_coco.py \
--cfg-options \
model.pretrained='focalnet_tiny_srf.pth' \
data.samples_per_gpu=2 \
model.backbone.focal_levels='[2,2,2,2]' \
--launcher pytorch
```

FocalNet-T (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/mask_rcnn_focalnet_tiny_patch4_mstrain_480-800_adamw_1x_coco.py \
--cfg-options \
model.pretrained='focalnet_tiny_lrf.pth' \
data.samples_per_gpu=2 \
model.backbone.focal_levels='[3,3,3,3]' \
--launcher pytorch
```

</details>

<details>

<summary>
Mask R-CNN with FocalNet-S
</summary>

FocalNet-S (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/mask_rcnn_focalnet_small_patch4_mstrain_480-800_adamw_1x_coco.py \
--cfg-options \
model.pretrained='focalnet_small_srf.pth' \
data.samples_per_gpu=2 \
model.backbone.focal_levels='[2,2,2,2]' \
--launcher pytorch
```

FocalNet-S (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/mask_rcnn_focalnet_small_patch4_mstrain_480-800_adamw_1x_coco.py \
--cfg-options \
model.pretrained='focalnet_small_lrf.pth' \
data.samples_per_gpu=2 \
model.backbone.focal_levels='[3,3,3,3]' \
--launcher pytorch
```

</details>

<details>

<summary>
Mask R-CNN with FocalNet-B
</summary>

FocalNet-B (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/mask_rcnn_focalnet_base_patch4_mstrain_480-800_adamw_1x_coco.py \
--cfg-options \
model.pretrained='focalnet_base_srf.pth' \
data.samples_per_gpu=2 \
model.backbone.focal_levels='[2,2,2,2]' \
--launcher pytorch
```

FocalNet-B (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/mask_rcnn_focalnet_base_patch4_mstrain_480-800_adamw_1x_coco.py \
--cfg-options \
model.pretrained='focalnet_base_lrf.pth' \
data.samples_per_gpu=2 \
model.backbone.focal_levels='[3,3,3,3]' \
--launcher pytorch
```

</details>

For training 3x models or other detection models, simply change the config file accordingly.