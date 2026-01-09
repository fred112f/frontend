import torch
import torch.nn as nn


class ANNClassifier(nn.Module):
    """Simple MLP for 48x48 grayscale images (7 classes)."""

    def __init__(
        self,
        num_classes: int = 7,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = 48 * 48  # grayscale 48x48

        layers: list[nn.Module] = []
        prev = input_dim

        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = h

        layers.append(nn.Linear(prev, num_classes))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 48, 48] -> [B, 2304]
        x = x.reshape(x.size(0), -1)
        return self.net(x)
