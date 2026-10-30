from torch import Tensor, nn

from decoder import Decoder
from encoder import Encoder


class Codec(nn.Module):
    def __init__(self, *, channel_count: int):
        super().__init__()

        self.encoder = Encoder(channel_count=channel_count)
        self.decoder = Decoder(channel_count=channel_count)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encoder(x)
        x = self.decoder(codes)
        return x, codes
