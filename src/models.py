import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import clip
from torchvision import transforms

class CLIPPretrainedModel(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128, dropout_rate: float = 0.5):
        super().__init__()
        
        # CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.clip_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        # EEG model
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, dropout_rate),
            ConvBlock(hid_dim, hid_dim, dropout_rate),
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim + 512, num_classes),  # CLIPの出力次元512を加える
            nn.Dropout(dropout_rate)
        )

    def forward(self, eeg_data: torch.Tensor, image_data: torch.Tensor) -> torch.Tensor:
        # EEG data through EEG model
        eeg_features = self.blocks(eeg_data)
        eeg_features = eeg_features.mean(dim=-1)
        
        # Image data through CLIP model
        image_data = self.transform(image_data).unsqueeze(0).to("cuda")
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_data)
        
        # Combine features
        combined_features = torch.cat((eeg_features, image_features), dim=-1)
        return self.head(combined_features)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate: float, kernel_size: int = 3, p_drop: float = 0.1):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)
