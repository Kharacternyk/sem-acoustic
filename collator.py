from torch import randint, stack
from torch.nn.functional import pad


class Collator:
    def __init__(self, *, segment_length):
        self.segment_length = segment_length

    def __call__(self, batch):
        collated_batch = []

        for waveform, *_ in batch:
            if waveform.size(-1) <= self.segment_length:
                waveform = pad(waveform, (0, self.segment_length - waveform.size(-1)))
            else:
                offset = randint(
                    0, waveform.size(-1) - self.segment_length + 1, ()
                ).item()
                waveform = waveform[..., offset : offset + self.segment_length]

            collated_batch.append(waveform)

        collated_batch = stack(collated_batch)

        return collated_batch
