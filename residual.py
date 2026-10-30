from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from snake import Snake


class ResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        channel_count: int,
        dilation: int,
    ):
        super().__init__()
        self.first_conv = weight_norm(
            nn.Conv1d(
                in_channels=channel_count,
                out_channels=channel_count,
                kernel_size=7,
                padding="same",
                dilation=dilation,
            )
        )
        self.second_conv = weight_norm(
            nn.Conv1d(
                in_channels=channel_count,
                out_channels=channel_count,
                kernel_size=1,
            )
        )
        self.first_snake = Snake(channel_count=channel_count)
        self.second_snake = Snake(channel_count=channel_count)

    def forward(self, x: Tensor) -> Tensor:
        y = self.first_snake(x)
        y = self.first_conv(y)
        y = self.second_snake(y)
        y = self.second_conv(y)
        y += x
        return y
