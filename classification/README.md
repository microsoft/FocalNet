# FocalNets for Image Classification

## Installation

Please follow [INSTALL.md](./INSTALL.md) for installation.

## Data preparation

Please following [DATA.md](./DATA.md) for data preparation.

## Evaluation

To evaluate a pre-trained FocalNets on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

For example, to evaluate the `FocalNet-B` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/focalnet_base_srf.yaml --resume focalnet_base_srf.pth --data-path <imagenet-path>
```

## Training from scratch

To train a FocalNet on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> --batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>
```

<details>

<summary>
FocalNet-T
</summary>
FocalNet-T (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/focalnet_tiny_srf.yaml --data-path <imagenet-path> --batch-size 128 
```

FocalNet-T (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/focalnet_tiny_lrf.yaml --data-path <imagenet-path> --batch-size 128 
```

</details>

<details>

<summary>
FocalNet-S
</summary>
FocalNet-S (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/focalnet_small_srf.yaml --data-path <imagenet-path> --batch-size 128 
```

FocalNet-S (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/focalnet_small_lrf.yaml --data-path <imagenet-path> --batch-size 128 
```

</details>

<details>

<summary>
FocalNet-B
</summary>
FocalNet-B (SRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/focalnet_base_srf.yaml --data-path <imagenet-path> --batch-size 128 
```

FocalNet-B (LRF):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/focalnet_base_lrf.yaml --data-path <imagenet-path> --batch-size 128 
```

</details>

## Throughput

To measure the throughput, run:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 128 --throughput --amp-opt-level O0
```

We reported the throughputs for our FocalNets on one V100 with batch size 128. 