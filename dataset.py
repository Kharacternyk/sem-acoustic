from csv import DictReader
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchaudio import load


class MAD(Dataset):
    sample_rate = 16_000
    categories = (
        "communication",
        "gunshot",
        "footsteps",
        "shelling",
        "vehicle",
        "helicopter",
        "fighter",
    )

    def __init__(self, root, *, split="training"):
        super().__init__()
        root = Path(root)

        self.samples = []
        with (root / f"{split}.csv").open(newline="") as metadata:
            for row in DictReader(metadata):
                path = root / row["path"]
                target = int(row["label"])
                self.samples.append((path, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        path, target = self.samples[index]
        waveform, sample_rate = load(path)
        assert sample_rate == self.sample_rate
        return waveform.mean(dim=0, keepdim=True), sample_rate, target
