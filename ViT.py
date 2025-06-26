import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import time
import numpy as np
from dadapy.data import Data    
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms.functional import InterpolationMode
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import handle_legacy_interface
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once

class ConvStemConfig(NamedTuple):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(hidden_dim, [mlp_dim, hidden_dim], activation_layer=nn.GELU, dropout=dropout)

    def forward(self, x):
        x = x + self.dropout(self.self_attention(self.ln_1(x), x, x, need_weights=False)[0])
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, hidden_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer) for _ in range(num_layers)]
        )
        self.norm = norm_layer(hidden_dim)

    def forward(self, x, num_blocks=None):
        x = x + self.pos_embedding
        x = self.dropout(x)
        num_blocks = num_blocks or len(self.layers)
        for layer in self.layers[:num_blocks]:
            x = layer(x)
        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.1, attention_dropout=0.1, num_classes=1000
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.conv_proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = Encoder(
            (image_size // patch_size) ** 2 + 1, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, partial(nn.LayerNorm, eps=1e-6)
        )
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def _process_input(self, x):
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        return x

    def forward(self, x, num_blocks=None):
        x = self._process_input(x)
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.encoder(x, num_blocks)
        if num_blocks:
            return x
        else:
            return self.head(x[:, 0])




def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}


class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_ops": 55.484,
            "_file_size": 331.398,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1



def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:

    weights = ViT_B_16_Weights.verify(weights)

    ### return _vision_transformer(
    #     patch_size=16,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=768,
    #     mlp_dim=3072,
    #     weights=weights,
    #     progress=progress,
    #     **kwargs,
    # )
    return _vision_transformer(
            patch_size=16,
            num_layers=6,
            num_heads=8,
            hidden_dim=int(768*0.678),
            mlp_dim=int(4*int(768*0.678)),
            weights=None,
            progress=progress,
            **kwargs,
        )

        
'''
#   TOY EXAMPLE
import torch

# Ensure model runs on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ViT model and move it to device
model = VisionTransformer(
    image_size=32, patch_size=4, num_layers=12, num_heads=4, hidden_dim=64, mlp_dim=128, num_classes=10
).to(device)

# Create dummy image tensor (batch size=1, 3 color channels, 32x32 image)
img = torch.randn(1, 3, 32, 32).to(device)

# Test progressive encoding
output_3_blocks = model(img, num_blocks=3)  # Output of 6 encoder blocks
output_12_blocks = model(img, num_blocks=12)  # Output of all 12 blocks

print("Output shape with 3 blocks:", output_3_blocks.shape)
#print("Output shape with 12 blocks:", output_12_blocks)
'''



'''
# using cifar datset

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match ViT input size
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
dataset = datasets.CIFAR10(root="/home/akshay/Intern_EE/data/cifar-10-batches-py./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get a batch of images
images, labels = next(iter(loader))

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = images.to(device)

# Initialize the Vision Transformer model
model = VisionTransformer(image_size=32, patch_size=4, num_layers=8, num_heads=4, hidden_dim=256, mlp_dim=256, num_classes=10).to(device)


import torch
import torchvision
import torchvision.transforms as transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load your ViT model and move it to GPU()   
model.to(device)


# Loop through dataset batches
ids=[] 

for l in range(8):
    # Store all outputs
    all_outputs = []
    for images, _ in loader:
        images = images.to(device)  # Move images to GPU

        with torch.no_grad():  # Disable gradient calculation to save memory
            batch_output = model(images, num_blocks=l+1)  # Model output (batch_size, 65, 256)

        all_outputs.append(batch_output)

    output = torch.cat(all_outputs, dim=0)  # Shape: (50000, 65, 256)
    output = output.mean(dim=1) # shape(50,000, 256)

    embed = output[torch.randperm(50000)[:200]]  # Randomly select 200 rows
# print(output.shape)
#print("output of l is : " , embed)
#print(embed.shape)


#embed is the embedding of shape- [Num_samples, EmbeddingSize]

    st2 = time.time()
    data_array = embed.detach().cpu().numpy()
    data_array = StandardScaler().fit_transform(data_array)
    data = Data(data_array)


    maxk_value = min(50, data.X.shape[0] - 1)  # Ensure maxk is valid
    #print(f"Using maxk={maxk_value}")

    id, _, _ = data.return_id_scaling_gride(range_max=130, d0=1, d1=70)

    en2 = time.time()
    ids.append(id)
    all_outputs.clear()

IDS = np.array(ids)

print("id you get for every l is:" ,IDS)
print(IDS.shape)
'''