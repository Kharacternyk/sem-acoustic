from torch import Tensor, nn

from residual import ResidualBlock
from snake import Snake


class Encoder(nn.Module):
    def __init__(self, *, channel_count: int):
        super().__init__()
        self.first_conv = nn.Conv1d(
            in_channels=1, out_channels=channel_count, kernel_size=7, padding="same"
        )
        self.final_conv = nn.Conv1d(
            in_channels=16 * channel_count,
            out_channels=16 * channel_count,
            kernel_size=3,
            padding="same",
        )
        self.final_snake = Snake(channel_count=16 * channel_count)

        self.project = nn.Conv1d(
            in_channels=16 * channel_count, out_channels=channel_count, kernel_size=1
        )
        self.project_snake = Snake(channel_count=16 * channel_count)

        self.first_block = EncoderBlock(channel_count=2 * channel_count, stride=2)
        self.second_block = EncoderBlock(channel_count=4 * channel_count, stride=4)
        self.third_block = EncoderBlock(channel_count=8 * channel_count, stride=5)
        self.fourth_block = EncoderBlock(channel_count=16 * channel_count, stride=8)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_conv(x)
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.fourth_block(x)
        x = self.final_snake(x)
        x = self.final_conv(x)

        x = self.project_snake(x)
        x = self.project(x)

        x = x.tanh().add(1).mul(4).sub(0.5)
        y = x.round().clamp(0, 7)
        x = x + y.sub(x).detach()
        x = x.div(3.5).sub(1)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, *, channel_count: int, stride: int):
        super().__init__()
        half_channel_count = channel_count // 2
        self.first_res = ResidualBlock(channel_count=half_channel_count, dilation=1)
        self.second_res = ResidualBlock(channel_count=half_channel_count, dilation=3)
        self.third_res = ResidualBlock(channel_count=half_channel_count, dilation=9)
        self.snake = Snake(channel_count=half_channel_count)
        self.conv = nn.Conv1d(
            in_channels=half_channel_count,
            out_channels=channel_count,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_res(x)
        x = self.second_res(x)
        x = self.third_res(x)
        x = self.snake(x)
        x = self.conv(x)
        return x
