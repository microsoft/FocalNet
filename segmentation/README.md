# FocalNets for Semantic Segmentation

## Installation and Data Preparation

We use [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and follow  [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) to set up our pipelines. 

## Evaluation

To evaluate a pre-trained FocalNets on ADE20K, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 tools/test.py \
<config-file> <ckpt-path> --options data.samples_per_gpu=<samples_per_gpu> --luancher pytorch --eval mIoU
```
For multi-scale evaluation, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 tools/test.py \
<config-file> <ckpt-path> --options data.samples_per_gpu=<samples_per_gpu> --luancher pytorch --eval mIoU --aug-test
```

For example, to evaluate the UperNet model with FocalNet-B (LRF) on 8 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 tools/test.py \
configs/focalnet/upernet_focalnet_base_patch4_512x512_160k_ade20k_lrf.py focalnet_base_lrf_upernet_160k.pth \
--cfg-options data.samples_per_gpu=1 model.backbone.focal_levels='[3,3,3,3]'
```

## Training

To train UperNet model with pretrained FocalNet, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  tools/train.py \
<config-file> \
--options \
model.pretrained=<pretrained/model/path> \
data.samples_per_gpu=<samples_per_gpu> \
--launcher pytorch
```

For example, we train UperNet with following commands:

<details>

<summary>
UperNet with FocalNet-T
</summary>

FocalNet-T (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/upernet_focalnet_tiny_patch4_512x512_160k_ade20k_srf.py \
--options \
model.pretrained='focalnet_tiny_srf.pth' \
data.samples_per_gpu=2 \
--launcher pytorch
```

FocalNet-T (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/upernet_focalnet_tiny_patch4_512x512_160k_ade20k_lrf.py \
--options \
model.pretrained='focalnet_tiny_lrf.pth' \
data.samples_per_gpu=2 \
--launcher pytorch
```

</details>

<details>

<summary>
UperNet with FocalNet-S
</summary>

FocalNet-S (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/upernet_focalnet_small_patch4_512x512_160k_ade20k_srf.py \
--options \
model.pretrained='focalnet_small_srf.pth' \
data.samples_per_gpu=2 \
--launcher pytorch
```

FocalNet-S (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/upernet_focalnet_small_patch4_512x512_160k_ade20k_lrf.py \
--options \
model.pretrained='focalnet_tiny_lrf.pth' \
data.samples_per_gpu=2 \
--launcher pytorch
```

</details>

<details>

<summary>
UperNet with FocalNet-B
</summary>

FocalNet-B (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/upernet_focalnet_base_patch4_512x512_160k_ade20k_srf.py \
--options \
model.pretrained='focalnet_base_srf.pth' \
data.samples_per_gpu=2 \
--launcher pytorch
```

FocalNet-B (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  tools/train.py \
configs/focalnet/upernet_focalnet_base_patch4_512x512_160k_ade20k_lrf.py \
--options \
model.pretrained='focalnet_base_lrf.pth' \
data.samples_per_gpu=2 \
--launcher pytorch
```

</details>