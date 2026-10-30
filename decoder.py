from torch import Tensor, nn

from residual import ResidualBlock
from snake import Snake


class Decoder(nn.Module):
    def __init__(self, *, channel_count: int):
        super().__init__()
        self.first_conv = nn.Conv1d(
            in_channels=16 * channel_count,
            out_channels=16 * channel_count,
            kernel_size=7,
            padding="same",
        )

        self.final_conv = nn.Conv1d(
            in_channels=channel_count, out_channels=1, kernel_size=7, padding="same"
        )
        self.final_snake = Snake(channel_count=channel_count)

        self.project = nn.Conv1d(
            in_channels=channel_count, out_channels=16 * channel_count, kernel_size=1
        )
        self.project_snake = Snake(channel_count=16 * channel_count)

        self.fourth_block = DecoderBlock(channel_count=2 * channel_count, stride=2)
        self.third_block = DecoderBlock(channel_count=4 * channel_count, stride=4)
        self.second_block = DecoderBlock(channel_count=8 * channel_count, stride=5)
        self.first_block = DecoderBlock(channel_count=16 * channel_count, stride=8)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project(x)
        x = self.project_snake(x)

        x = self.first_conv(x)
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.fourth_block(x)
        x = self.final_snake(x)
        x = self.final_conv(x)
        x = x.tanh()

        return x


class DecoderBlock(nn.Module):
    def __init__(self, *, channel_count: int, stride: int):
        super().__init__()
        self.first_res = ResidualBlock(channel_count=channel_count // 2, dilation=1)
        self.second_res = ResidualBlock(channel_count=channel_count // 2, dilation=3)
        self.third_res = ResidualBlock(channel_count=channel_count // 2, dilation=9)
        self.conv = nn.ConvTranspose1d(
            in_channels=channel_count,
            out_channels=channel_count // 2,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride,
        )
        self.snake = Snake(channel_count=channel_count)

    def forward(self, x: Tensor) -> Tensor:
        x = self.snake(x)
        x = self.conv(x)
        x = self.first_res(x)
        x = self.second_res(x)
        x = self.third_res(x)
        return x
