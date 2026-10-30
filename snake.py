from torch import Tensor, distributions, jit, nn


class Snake(nn.Module):
    def __init__(self, *, channel_count: int):
        super().__init__()
        self.factor = nn.Parameter(
            distributions.Exponential(1).sample((1, channel_count, 1))
        )

    def forward(self, x: Tensor) -> Tensor:
        return snake(x, self.factor)


@jit.script
def snake(x: Tensor, factor: Tensor) -> Tensor:
    return x + x.mul(factor).sin().pow(2).divide(factor + 1e-8)
