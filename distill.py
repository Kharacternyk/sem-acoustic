from collections import defaultdict
from itertools import count
from pathlib import Path

import modal

from checkpoint import find_checkpoint
from modal_train import checkpoints, datasets, image

app = modal.App("sem-acoustic-distillation")

segment_length = 7_950
accumulation_steps = 1
batch_size = 32
encoder_channel_count = 8


def get_lr(step: int) -> float:
    return 2e-4 * 0.5 ** (step / 30_000)


def load_models(teacher_checkpoint: Path, checkpoint: Path | None = None):
    from torch import load

    from codec import Codec
    from encoder import Encoder

    model = Codec(channel_count=64)
    model.load_state_dict(
        load(teacher_checkpoint, map_location="cpu", weights_only=True)
    )
    model.requires_grad_(False).eval()
    teacher = model.encoder
    model.encoder = Encoder(
        channel_count=encoder_channel_count,
        code_channel_count=64,
    )

    if checkpoint is not None:
        state = load(checkpoint, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(
            {
                name.removeprefix("encoder."): value
                for name, value in state.items()
                if name.startswith("encoder.")
            }
        )

    model.train()
    model.decoder.eval()
    return teacher, model


def train(teacher_checkpoint: Path, directory: Path, resume_index: int):
    from torch import (
        backends,
        inference_mode,
        load,
        no_grad,
        save,
        set_float32_matmul_precision,
        stack,
        zeros_like,
    )
    from torch.compiler import cudagraph_mark_step_begin
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import wandb
    from collator import Collator
    from dataset import MAD
    from losses import DistillationLoss, ReconstructionLoss

    dataset = MAD(Path("/datasets/mad"))
    val_dataset = MAD(Path("/datasets/mad"), split="test")
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

    checkpoint = directory / f"{resume_index:06}"
    teacher, model = load_models(
        teacher_checkpoint, checkpoint / "model.pt" if resume_index else None
    )
    teacher.cuda()
    model.cuda()
    teacher.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
    model.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")

    lr = get_lr(resume_index)
    optimizer = AdamW(model.encoder.parameters(), lr, (0.8, 0.9), fused=True)

    if resume_index:
        optimizer.load_state_dict(load(checkpoint / "optimizer.pt", weights_only=True))

        for p in optimizer.param_groups:
            p["lr"] = lr

    get_rec_loss = ReconstructionLoss(sample_rate=dataset.sample_rate).cuda()
    get_distillation_loss = DistillationLoss().cuda()

    with wandb.init(
        entity="kharacternyk-team", project="sem-acoustic-distill"
    ) as logger:
        metrics = defaultdict(list)
        global_index = resume_index
        accumulation_index = 0

        for p in model.encoder.parameters():
            p.grad = zeros_like(p)

        for _ in count():
            for batch in tqdm(dataloader):
                cudagraph_mark_step_begin()
                batch = batch.cuda(non_blocking=True)

                with no_grad():
                    teacher_codes = teacher(batch).clone()

                output, codes = model(batch)
                output = output.clone()
                codes = codes.clone()

                rec_loss = get_rec_loss(batch, output)
                distillation_loss = get_distillation_loss(teacher_codes, codes)
                total_loss = rec_loss + distillation_loss
                (total_loss / accumulation_steps).backward()

                batch_metrics = dict(
                    total_loss=total_loss,
                    rec_loss=rec_loss,
                    distillation_loss=distillation_loss,
                    code_sample_std=codes.detach()
                    .mean(dim=-1)
                    .std(dim=0, unbiased=False)
                    .mean(),
                    code_temporal_std=codes.detach().std(dim=-1, unbiased=False).mean(),
                )

                for name, value in batch_metrics.items():
                    metrics[name].append(value.detach())

                accumulation_index += 1
                if accumulation_index % accumulation_steps:
                    continue

                optimizer.step()
                optimizer.zero_grad(set_to_none=False)

                global_index += 1
                lr = get_lr(global_index)

                for p in optimizer.param_groups:
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
                    checkpoint = directory / f"{global_index:06}"
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    save(model.state_dict(), checkpoint / "model.pt")
                    save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                    checkpoints.commit()

                    model.encoder.eval()
                    rec_losses = []
                    distillation_losses = []

                    with inference_mode():
                        for batch in tqdm(val_dataloader):
                            cudagraph_mark_step_begin()
                            batch = batch.cuda(non_blocking=True)
                            teacher_codes = teacher(batch).clone()
                            output, codes = model(batch)
                            output = output.clone()
                            codes = codes.clone()
                            rec_losses.append(get_rec_loss(batch, output))
                            distillation_losses.append(
                                get_distillation_loss(teacher_codes, codes)
                            )

                        rec_loss = stack(rec_losses).mean()
                        distillation_loss = stack(distillation_losses).mean()
                        logger.log(
                            {
                                "val/rec_loss": rec_loss,
                                "val/distillation_loss": distillation_loss,
                                "val/total_loss": rec_loss + distillation_loss,
                            },
                            step=global_index,
                        )

                    model.encoder.train()


@app.function(
    image=image,
    gpu="L40S",
    cpu=1.5,
    memory=1024 * 8,
    timeout=24 * 60 * 60,
    retries=3,
    volumes={"/checkpoints": checkpoints, "/datasets": datasets},
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    include_source=False,
)
def run(teacher_run: str, run_name: str = "default", teacher_index: int = 0):
    checkpoints.reload()
    datasets.reload()
    directory = Path("/checkpoints/distilled") / run_name
    directory.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint = find_checkpoint(
        Path("/checkpoints") / teacher_run, teacher_index or None
    )
    checkpoint = find_checkpoint(directory, None, files=("model.pt", "optimizer.pt"))
    resume_index = int(checkpoint.parent.name)

    try:
        train(teacher_checkpoint, directory, resume_index)
    finally:
        checkpoints.commit()


@app.local_entrypoint()
def main(teacher_run: str, run_name: str = "default", teacher_index: int = 0):
    run.spawn(
        teacher_run=teacher_run, run_name=run_name, teacher_index=teacher_index
    ).get()
