import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("PREVAD_DATA_ROOT", REPO_ROOT / "datasets"))
HF_CACHE_DIR = os.getenv("PREVAD_HF_CACHE", str(REPO_ROOT / "huggingface_hub"))
EXTERNAL_ROOT = Path(os.getenv("PREVAD_EXTERNAL_ROOT", REPO_ROOT / "external"))
OUTPUT_ROOT = Path(os.getenv("PREVAD_OUTPUT_ROOT", REPO_ROOT))


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def data_path(*parts: str) -> str:
    return str(DATA_ROOT.joinpath(*parts))


def external_path(*parts: str) -> str:
    return str(EXTERNAL_ROOT.joinpath(*parts))


def output_path(*parts: str) -> str:
    return str(OUTPUT_ROOT.joinpath(*parts))
