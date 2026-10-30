from pathlib import Path

import modal

from checkpoint import find_checkpoint

root = Path(__file__).parent
remote_root = Path("/root/sem-acoustic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "ffmpeg")
    .pip_install(
        "torch==2.13.0",
        "torchaudio==2.11.0",
        "torchcodec==0.17.0",
        "encodec==0.1.1",
        "wandb==0.31.0",
        "tqdm==4.70.0",
    )
    .workdir(str(remote_root))
)

for source in root.glob("*.py"):
    image = image.add_local_file(source, str(remote_root / source.name))

app = modal.App("sem-acoustic")
datasets = modal.Volume.from_name("datasets", create_if_missing=True, version=2)
checkpoints = modal.Volume.from_name(
    "sem-acoustic-checkpoints", create_if_missing=True, version=2
)


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
def run(run_name: str):
    from train import train

    checkpoints.reload()
    datasets.reload()
    directory = Path("/checkpoints") / run_name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = find_checkpoint(directory, None)
    resume_index = int(checkpoint.parent.name)

    try:
        train(
            dataset_root=Path("/datasets/mad"),
            checkpoints=Path("/checkpoints"),
            run_name=run_name,
            resume_index=resume_index,
            on_checkpoint=checkpoints.commit,
        )
    finally:
        checkpoints.commit()


@app.local_entrypoint()
def main(run_name: str):
    run.spawn(run_name=run_name).get()
