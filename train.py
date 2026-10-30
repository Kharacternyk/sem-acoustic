from collections import defaultdict
from collections.abc import Callable
from itertools import count
from pathlib import Path

from encodec.msstftd import DiscriminatorSTFT
from torch import (
    backends,
    inference_mode,
    load,
    nn,
    save,
    set_float32_matmul_precision,
    stack,
    zeros_like,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from codec import Codec
from collator import Collator
from dataset import MAD
from losses import (
    ReconstructionLoss,
    get_d_loss,
    get_feat_loss,
    get_g_loss,
)
from period_discriminator import PeriodDiscriminator
from scale_discriminator import ScaleDiscriminator

segment_length = 7_950
accumulation_steps = 1
batch_size = 32


def get_lr(step: int) -> float:
    return 1e-4 * 0.5 ** (step / 30_000)


def train(
    *,
    dataset_root: Path = Path("mad"),
    checkpoints: Path = Path("checkpoints"),
    run_name: str = "default",
    resume_index: int = 0,
    on_checkpoint: Callable[[], None] | None = None,
):
    checkpoints = checkpoints / run_name
    dataset = MAD(dataset_root)
    val_dataset = MAD(dataset_root, split="test")

    collator = Collator(segment_length=segment_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collator,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collator,
        drop_last=True,
    )

    backends.cudnn.benchmark = True
    set_float32_matmul_precision("medium")

    lr = get_lr(resume_index)

    model = Codec(channel_count=64).cuda()

    if resume_index:
        model.load_state_dict(load(checkpoints / f"{resume_index:06}" / "model.pt"))

    model.encoder.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
    model.decoder.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
    model.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")

    optimizer = AdamW(model.parameters(), lr, (0.8, 0.9), fused=True)

    if resume_index:
        optimizer.load_state_dict(
            load(checkpoints / f"{resume_index:06}" / "optimizer.pt")
        )

        for p in optimizer.param_groups:
            p["lr"] = lr

    get_rec_loss = ReconstructionLoss(sample_rate=dataset.sample_rate).cuda()

    discriminators = nn.ModuleList(
        [
            PeriodDiscriminator(period=2),
            PeriodDiscriminator(period=3),
            PeriodDiscriminator(period=5),
            PeriodDiscriminator(period=7),
            PeriodDiscriminator(period=11),
            ScaleDiscriminator(sample_rate=dataset.sample_rate, downsample_rate=2),
            ScaleDiscriminator(sample_rate=dataset.sample_rate, downsample_rate=4),
            DiscriminatorSTFT(32, n_fft=2048, win_length=2048, hop_length=512),
            DiscriminatorSTFT(32, n_fft=1024, win_length=1024, hop_length=256),
            DiscriminatorSTFT(32, n_fft=512, win_length=512, hop_length=128),
            DiscriminatorSTFT(32, n_fft=256, win_length=256, hop_length=64),
            DiscriminatorSTFT(32, n_fft=128, win_length=128, hop_length=32),
        ]
    ).cuda()

    if resume_index:
        discriminators.load_state_dict(
            load(checkpoints / f"{resume_index:06}" / "discriminators.pt")
        )

    d_optimizer = AdamW(discriminators.parameters(), lr, (0.8, 0.9), fused=True)

    if resume_index:
        d_optimizer.load_state_dict(
            load(checkpoints / f"{resume_index:06}" / "d_optimizer.pt")
        )

        for p in d_optimizer.param_groups:
            p["lr"] = lr

    with wandb.init("kharacternyk-team", project="sem-acoustic") as logger:
        metrics = defaultdict(list)
        global_index = resume_index
        accumulation_index = 0
        for module in (model, discriminators):
            for p in module.parameters():
                p.grad = zeros_like(p)

        for _ in count():
            for batch in tqdm(dataloader):
                batch = batch.cuda(non_blocking=True)

                output, codes = model(batch)
                output = output.clone()
                codes = codes.clone()

                for p in discriminators.parameters():
                    p.requires_grad_(True)

                d_losses = []

                for discriminator in discriminators:
                    d_losses.append(get_d_loss(batch, output, discriminator))

                (stack(d_losses).mean() / accumulation_steps).backward()

                for p in discriminators.parameters():
                    p.requires_grad_(False)

                g_losses = []
                feat_losses = []

                for discriminator in discriminators:
                    g_losses.append(get_g_loss(output, discriminator))
                    feat_losses.append(get_feat_loss(batch, output, discriminator))

                adv_loss = stack(g_losses + feat_losses).mean()
                rec_loss = get_rec_loss(batch, output)

                total_loss = adv_loss + rec_loss
                (total_loss / accumulation_steps).backward()

                batch_metrics = dict(
                    total_loss=total_loss,
                    rec_loss=rec_loss,
                    adv_loss=adv_loss,
                    code_sample_std=codes.detach()
                    .mean(dim=-1)
                    .std(dim=0, unbiased=False)
                    .mean(),
                    code_temporal_std=codes.detach().std(dim=-1, unbiased=False).mean(),
                )

                for i in range(len(discriminators)):
                    batch_metrics.update(
                        {
                            f"d_loss/{i}": d_losses[i],
                            f"g_loss/{i}": g_losses[i],
                            f"feat_loss/{i}": feat_losses[i],
                        }
                    )

                for name, value in batch_metrics.items():
                    metrics[name].append(value.detach())

                accumulation_index += 1
                if accumulation_index % accumulation_steps:
                    continue

                d_optimizer.step()
                optimizer.step()
                d_optimizer.zero_grad(set_to_none=False)
                optimizer.zero_grad(set_to_none=False)

                global_index += 1
                lr = get_lr(global_index)

                for opt in (optimizer, d_optimizer):
                    for p in opt.param_groups:
                        p["lr"] = lr

                if not global_index % 50:
                    logger.log(
                        {
                            "lr": lr,
                            **{
                                name: stack(values).mean()
                                for name, values in metrics.items()
                            },
                        },
                        step=global_index,
                    )
                    metrics.clear()

                if not global_index % 500:
                    checkpoint = checkpoints / f"{global_index:06}"
                    checkpoint.mkdir(parents=True, exist_ok=True)

                    save(model.state_dict(), checkpoint / "model.pt")
                    save(discriminators.state_dict(), checkpoint / "discriminators.pt")
                    save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                    save(d_optimizer.state_dict(), checkpoint / "d_optimizer.pt")

                    if on_checkpoint is not None:
                        on_checkpoint()

                    model.eval()
                    discriminators.eval()

                    rec_losses = []
                    adv_losses = []

                    with inference_mode():
                        for batch in tqdm(val_dataloader):
                            g_losses = []
                            feat_losses = []
                            batch = batch.cuda(non_blocking=True)

                            output, _ = model(batch)
                            output = output.clone()

                            for discriminator in discriminators:
                                g_losses.append(get_g_loss(output, discriminator))
                                feat_losses.append(
                                    get_feat_loss(batch, output, discriminator)
                                )

                            rec_losses.append(get_rec_loss(batch, output))
                            adv_losses.append(stack(g_losses + feat_losses).mean())

                        rec_loss = stack(rec_losses).mean()
                        adv_loss = stack(adv_losses).mean()
                        total_loss = rec_loss + adv_loss

                        logger.log(
                            {
                                "val/rec_loss": rec_loss,
                                "val/adv_loss": adv_loss,
                                "val/total_loss": total_loss,
                            },
                            step=global_index,
                        )

                    model.train()
                    discriminators.train()
