from functools import reduce

from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Resample


class ScaleDiscriminator(nn.Module):
    def __init__(self, *, sample_rate: int, downsample_rate: int):
        super().__init__()
        self.resample = (
            Resample(sample_rate, sample_rate // downsample_rate)
            if downsample_rate > 1
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                get_layer(1, 16, kernel_size=15, stride=1, padding=7),
                get_layer(16, 64, kernel_size=41, stride=4, groups=4, padding=20),
                get_layer(64, 256, kernel_size=41, stride=4, groups=16, padding=20),
                get_layer(256, 1024, kernel_size=41, stride=4, groups=64, padding=20),
                get_layer(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20),
                get_layer(1024, 1024, kernel_size=5, stride=1, padding=2),
                get_layer(1024, 1, kernel_size=3, stride=1, padding=1),
            ]
        )

    def forward(self, x: Tensor):
        x = self.resample(x)

        features = reduce(
            lambda features, layer: features + [layer(features[-1])],
            self.layers,
            [x],
        )

        return features[-1], features[1:-1]


def get_layer(input_channel_count, output_channel_count, **kwargs):
    layer = weight_norm(nn.Conv1d(input_channel_count, output_channel_count, **kwargs))
    if output_channel_count == 1:
        return layer
    return nn.Sequential(layer, nn.LeakyReLU(0.1))
