from torch import Tensor, nn, stack
from torchaudio.transforms import MelSpectrogram


class ReconstructionLoss(nn.Module):
    def __init__(self, *, sample_rate: int):
        super().__init__()

        self.transforms = nn.ModuleList()

        n_fft = 32
        hop_length = 8
        n_mels = 5

        for _ in range(7):
            transform = MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            self.transforms.append(transform)

            n_fft *= 2
            hop_length *= 2
            n_mels *= 2

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        l1s = []
        l2s = []

        for melspec in self.transforms:
            spec_x = melspec(x)
            spec_y = melspec(y)

            if spec_x.size(-1) > spec_y.size(-1):
                spec_x = spec_x[..., : spec_y.size(-1)]
            elif spec_x.size(-1) < spec_y.size(-1):
                spec_y = spec_y[..., : spec_x.size(-1)]

            l1s.append(spec_x.sub(spec_y).abs().mean())

            factor = melspec.n_fft
            factor /= 2
            factor **= 0.5

            epsilon = 1e-5
            spec_x = spec_x.add(epsilon).log()
            spec_y = spec_y.add(epsilon).log()

            l2s.append(spec_x.sub(spec_y).square().mean().mul(factor))

        l1 = stack(l1s).mean()
        l2 = stack(l2s).mean()

        return l1 + l2


def get_d_loss(batch, output, discriminator):
    output = output.detach()

    x = discriminator(output)[0]
    x = x.add(1).relu().mean()

    y = discriminator(batch)[0]
    y = y.neg().add(1).relu().mean()

    return x + y


def get_g_loss(output, discriminator):
    x = discriminator(output)[0]

    return x.neg().add(1).relu().mean()


def get_feat_loss(batch, output, discriminator):
    xs = discriminator(batch)[1]
    ys = discriminator(output)[1]

    losses = [x.detach().sub(y).abs().mean() for x, y in zip(xs, ys)]

    return stack(losses).mean().mul(100)
