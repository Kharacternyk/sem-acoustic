from functools import reduce

from torch import Tensor, nn
from torch.nn.functional import pad
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, *, period: int):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                get_layer(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                get_layer(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                get_layer(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                get_layer(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                get_layer(1024, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0)),
                get_layer(1024, 1, kernel_size=(3, 1), padding=(1, 0)),
            ]
        )

    def forward(self, x: Tensor):
        x = pad(x, (0, (-x.size(-1)) % self.period), mode="reflect")
        batch_size, channel_count, length = x.shape
        x = x.reshape(batch_size, channel_count, length // self.period, self.period)

        features = reduce(
            lambda features, layer: features + [layer(features[-1])],
            self.layers,
            [x],
        )

        return features[-1], features[1:-1]


def get_layer(input_channel_count, output_channel_count, **kwargs):
    layer = weight_norm(nn.Conv2d(input_channel_count, output_channel_count, **kwargs))
    if output_channel_count == 1:
        return layer
    return nn.Sequential(layer, nn.LeakyReLU(0.1))
