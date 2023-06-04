# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from .llms.in1k_vocab import IMAGENET_CLASSES
from .llms.llama_modeling import LlamaForCausalLM
from .focalnet import FocalNet

class VisionLLM(nn.Module):
    r""" Focal Modulation Networks (FocalNets) with LLMs

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_postln (bool): Whether use layernorm after modulation (it helps stablize training of large models)
    """
    def __init__(self, vision_model):
        super().__init__()

        # build vision model
        self.vision = vision_model
        # fix head
        for param in self.vision.parameters():
            param.requires_grad = False

        # build adapter
        self.adapter = nn.Linear(self.vision.num_features, 4096)
        self.adapter.weight.data.fill_(0.0)
        self.adapter.bias.data.fill_(0.0)

        # build llm
        self.llama = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float16)
        for param in self.llama.parameters():
            param.requires_grad = False

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf") # from_pretrained("huggyllama/llama-7b") # AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        class_tokenized = self.tokenizer(["{}.".format(c.lower()) for c in IMAGENET_CLASSES], return_tensors="pt", add_special_tokens=False, padding=True)
        self.register_buffer("class_input_ids", class_tokenized["input_ids"])
        self.register_buffer("class_attn_masks", class_tokenized["attention_mask"])

        prefix_tokenized = self.tokenizer("<image>", return_tensors="pt", add_special_tokens=False)
        self.register_buffer("prefix_input_ids", prefix_tokenized["input_ids"])
        self.register_buffer("prefix_attn_masks", prefix_tokenized["attention_mask"])

        suffix_tokenized = self.tokenizer("</image> Read the image carefully, and select the correct image category from three <category, score> pairs:", return_tensors="pt", add_special_tokens=False)
        self.register_buffer("suffix_input_ids", suffix_tokenized["input_ids"])
        self.register_buffer("suffix_attn_masks", suffix_tokenized["attention_mask"])


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {''}


    def forward(self, x, targets=None, history=None, instruct=". Answer:", max_length=10):
        # with torch.cuda.amp.autocast():
        x = self.vision.forward_featuremaps(x)    
        pooled_x = self.vision.avgpool(x.transpose(1, 2))  # B C 1
        pooled_x = torch.flatten(pooled_x, 1)
        scores = self.vision.head(pooled_x)
        probs = torch.softmax(scores, 1)
        acc_top1 = (scores.argmax(1) == targets).sum() / targets.shape[0]
        acc_top3 = (scores.sort(1, descending=True)[1] == targets.view(-1, 1))[:, :3].sum() / targets.shape[0]

        # find the top candidates
        certain_inds = []
        certain_decoded = []
        uncertain_inds = []
        candidate_str = []
        probs_sorted, order_sorted = probs.sort(1, descending=True)
        for i in range(scores.shape[0]):
            if probs_sorted[i][0] < 0.5:
                candidate_str_i = []
                for k in range(3):
                    candidate_str_i.append("category: {}, score: {:.2f}".format(IMAGENET_CLASSES[order_sorted[i][k].item()], probs_sorted[i][k].item()))
                candidate_str_i = "; ".join(candidate_str_i)
                candidate_str.append(candidate_str_i + '. Answer:')
                uncertain_inds.append(i)
            else:
                certain_inds.append(i)
                certain_decoded.append(IMAGENET_CLASSES[order_sorted[i][0].item()])

        certain_targets = targets[torch.LongTensor(certain_inds).to(x.device)]      

        uncertain_inds = torch.LongTensor(uncertain_inds).to(x.device)
        uncertain_x = x[uncertain_inds]
        uncertain_targets = targets[uncertain_inds]

        uncertain_x = self.adapter(uncertain_x)  # B L 4096
        if self.training:
            visual_tokens = uncertain_x
            
            prefix_textual_tokens = self.llama.model.embed_tokens(self.prefix_input_ids)
            prefix_attn_masks = self.prefix_attn_masks

            suffix_textual_tokens = self.llama.model.embed_tokens(self.suffix_input_ids)
            suffix_attn_masks = self.suffix_attn_masks

            # tokenize options
            option_tokenized = self.tokenizer(candidate_str, return_tensors="pt", add_special_tokens=False, padding=True)
            option_input_ids = option_tokenized["input_ids"].to(x.device)
            option_attention_mask = option_tokenized["attention_mask"].to(x.device)
            option_textual_tokens = self.llama.model.embed_tokens(option_input_ids)

            target_input_ids = self.class_input_ids[uncertain_targets]
            target_attn_masks = self.class_attn_masks[uncertain_targets]
            target_textual_tokens = self.llama.model.embed_tokens(target_input_ids)

            tokens = torch.cat((prefix_textual_tokens.repeat(uncertain_x.shape[0], 1, 1), visual_tokens, suffix_textual_tokens.repeat(uncertain_x.shape[0], 1, 1), option_textual_tokens, target_textual_tokens), 1)          
            attn_visual = torch.ones(visual_tokens.shape[:-1], dtype=torch.long).to(visual_tokens.device)            
            attention_mask = torch.cat([prefix_attn_masks.repeat(uncertain_x.shape[0], 1), attn_visual, suffix_attn_masks.repeat(uncertain_x.shape[0], 1), option_attention_mask, target_attn_masks], dim=1)            

            with torch.cuda.amp.autocast():
                outputs = self.llama(
                    inputs_embeds=tokens, 
                    attention_mask=attention_mask,
                    labels=target_input_ids
                )
            loss = outputs.loss
            return loss
        else:
            visual_tokens = uncertain_x

            prefix_textual_tokens = self.llama.model.embed_tokens(self.prefix_input_ids)
            prefix_attn_masks = self.prefix_attn_masks

            suffix_textual_tokens = self.llama.model.embed_tokens(self.suffix_input_ids)
            suffix_attn_masks = self.suffix_attn_masks

            inst_tokenized = self.tokenizer(instruct, return_tensors="pt")
            inst_input_ids = inst_tokenized["input_ids"].to(x.device)
            inst_attention_mask = inst_tokenized["attention_mask"].to(x.device)
            inst_textual_tokens = self.llama.model.embed_tokens(inst_input_ids)
            
            uncertain_decoded = []
            for k in range(len(candidate_str)):
                # tokenize options
                option_tokenized = self.tokenizer(candidate_str[k:k+1], return_tensors="pt", add_special_tokens=False, padding=False)
                option_input_ids = option_tokenized["input_ids"].to(x.device)
                option_attention_mask = option_tokenized["attention_mask"].to(x.device)
                option_textual_tokens = self.llama.model.embed_tokens(option_input_ids)

                tokens = torch.cat((prefix_textual_tokens, visual_tokens[k:k+1], suffix_textual_tokens, option_textual_tokens), 1)          
                attn_visual = torch.ones(visual_tokens[k:k+1].shape[:-1], dtype=torch.long).to(visual_tokens.device)            
                attention_mask = torch.cat([prefix_attn_masks, attn_visual, suffix_attn_masks, option_attention_mask], dim=1)            

                with torch.cuda.amp.autocast():
                    generated = self.llama.generate(
                        inputs_embeds=tokens, 
                        attention_mask=attention_mask,
                        max_length=max_length
                    )
                decoded_k = self.tokenizer.decode(generated[0])
                uncertain_decoded.append(decoded_k)

            decoded_all = certain_decoded + uncertain_decoded
            target_all = torch.cat((certain_targets, uncertain_targets), 0)

            return decoded_all, target_all, acc_top1 # (history_tokens, history_attn_masks)

    def flops(self):
        return self.vision.flops()

def build_transforms(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation='bicubic')
        )
        t.append(
            transforms.CenterCrop(img_size)    
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation='bicubic')
        )        
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_transforms4display(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation='bicubic')
        )
        t.append(
            transforms.CenterCrop(img_size)    
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation='bicubic')
        )  
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

model_urls = {
    "focalnet_tiny_srf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_srf.pth",
    "focalnet_tiny_lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_lrf.pth",
    "focalnet_small_srf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_srf.pth",
    "focalnet_small_lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_lrf.pth",
    "focalnet_base_srf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_srf.pth",
    "focalnet_base_lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_lrf.pth",    
    "focalnet_large_fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384.pth", 
    "focalnet_large_fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384_fl4.pth", 
    "focalnet_xlarge_fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384.pth", 
    "focalnet_xlarge_fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384_fl4.pth", 
    "focalnet_huge_fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_huge_lrf_224.pth", 
    "focalnet_huge_fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_huge_lrf_224_fl4.pth", 
}

@register_model
def focalnet_tiny_srf_llm(pretrained=False, **kwargs):
    vision = FocalNet(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    if pretrained:
        vision.load_state_dict(torch.load("focalnet_tiny_srf.pth")["model"])
        # url = model_urls['focalnet_tiny_srf']
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        # model.load_state_dict(checkpoint["model"])
    model = VisionLLM(vision)
    return model

@register_model
def focalnet_small_srf_llm(pretrained=False, **kwargs):
    model = FocalNetLLM(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    if pretrained:
        url = model_urls['focalnet_small_srf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_base_srf_llm(pretrained=False, **kwargs):
    model = FocalNetLLM(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    if pretrained:
        url = model_urls['focalnet_base_srf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == '__main__':
    img_size = 224
    x = torch.rand(16, 3, img_size, img_size).cuda()
    model = FocalNetLLM(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3]).cuda()
    print(model); 
    model(x)

    flops = model.flops()
    print(f"number of GFLOPs: {flops / 1e9}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
