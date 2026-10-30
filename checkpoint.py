from pathlib import Path


def find_checkpoint(directory: Path, index: int | None) -> Path:
    if index is None:
        files = ("model.pt", "discriminators.pt", "optimizer.pt", "d_optimizer.pt")
        index = max(
            (
                int(path.name)
                for path in directory.iterdir()
                if path.is_dir()
                and path.name.isdecimal()
                and all((path / name).is_file() for name in files)
            ),
            default=0,
        )

    return directory / f"{index:06}" / "model.pt"
