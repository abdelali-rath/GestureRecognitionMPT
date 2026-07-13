import os
from pathlib import Path
from typing import Iterable


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATASET_ENV_VAR = "GESTURE_DATASET_DIR"


def _resolve_relative(path: Path, bases: Iterable[Path], fallback_base: Path) -> Path:
    if path.is_absolute():
        return path

    for base in bases:
        candidate = (base / path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    return (fallback_base / path).expanduser().resolve()


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a path relative to the project root, independent of the cwd."""
    requested = Path(path).expanduser()
    if requested.is_absolute():
        return requested

    return (PROJECT_DIR / requested).resolve()


def resolve_dataset_dir(base_path: str | os.PathLike[str] = "dataset") -> Path:
    """Return the dataset directory.

    Relative dataset paths are resolved against the repository root first. This
    keeps scripts working whether they are launched from the project root, from
    the package folder, or from an IDE. Set GESTURE_DATASET_DIR to override the
    default dataset location.
    """
    env_path = os.environ.get(DATASET_ENV_VAR)
    if env_path and Path(base_path) == Path("dataset"):
        return _resolve_relative(Path(env_path).expanduser(), [Path.cwd()], PROJECT_DIR)

    requested = Path(base_path).expanduser()
    return _resolve_relative(requested, [PROJECT_DIR, PACKAGE_DIR, Path.cwd()], PROJECT_DIR)


def resolve_label_dir(
    label: str,
    base_path: str | os.PathLike[str] = "dataset",
    *,
    create: bool = False,
) -> Path:
    """Resolve a label folder and tolerate case differences between systems."""
    label_name = str(label).strip()
    if not label_name:
        raise ValueError("Label darf nicht leer sein.")

    dataset_dir = resolve_dataset_dir(base_path)
    exact_dir = dataset_dir / label_name

    if exact_dir.exists():
        return exact_dir.resolve()

    if dataset_dir.exists():
        label_key = label_name.casefold()
        for child in dataset_dir.iterdir():
            if child.is_dir() and child.name.casefold() == label_key:
                return child.resolve()

    if create:
        exact_dir.mkdir(parents=True, exist_ok=True)

    return exact_dir.resolve()


def iter_npy_files(folder: str | os.PathLike[str]) -> list[Path]:
    path = Path(folder)
    if not path.exists():
        return []

    return sorted(
        (child.resolve() for child in path.iterdir() if child.is_file() and child.suffix.casefold() == ".npy"),
        key=lambda child: child.name.casefold(),
    )
