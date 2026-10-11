import timm
import torch
from torch import nn
from torchvision.transforms import v2

__all__ = ["GeoModel", "train_tf", "val_tf"]

IMG_SIZE = 224

train_tf = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.ColorJitter(0.3, 0.3, 0.3),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_tf = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.CenterCrop(IMG_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class GeoModel(nn.Module):
    def __init__(
        self,
        map_size: int,
        hidden_size: int = 1024,
        dropout: float = 0.1,
        images_per_loc: int = 2,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            f"tiny_vit_5m_{IMG_SIZE}.dist_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
        )
        feature_dim = int(self.encoder.num_features)  # ty: ignore

        self.head = nn.Sequential(
            nn.Linear(feature_dim * images_per_loc, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, map_size),
        )

    def forward(self, images: torch.Tensor):
        batched = torch.flatten(images, start_dim=0, end_dim=1)
        encoded = self.encoder(batched)
        encoded = encoded.view(images.shape[0], -1)

        return self.head(encoded)
