from torch import Tensor, arange, nn


class Classifier(nn.Module):
    def __init__(self, *, channel_count: int, class_count: int):
        super().__init__()
        self.channel_dropout = nn.Dropout1d(0.2)
        self.first_conv = nn.Conv1d(channel_count, 32, kernel_size=3, padding=1)
        self.second_conv = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.2)
        self.project = nn.Linear(32, class_count)

    def forward(self, codes: Tensor, lengths: Tensor) -> Tensor:
        padding = (
            arange(codes.size(-1), device=codes.device)[None] >= lengths[:, None]
        ).unsqueeze(1)
        codes = codes.masked_fill(padding, 0)

        x = self.channel_dropout(codes)
        x = self.activation(self.first_conv(x))
        x = x.masked_fill(padding, 0)
        x = self.activation(self.second_conv(x))
        x = x.masked_fill(padding, 0).sum(dim=-1) / lengths[:, None]

        return self.project(self.dropout(x))
