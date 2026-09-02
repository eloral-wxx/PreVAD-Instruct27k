import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKS = [
    (
        "PreVAD-Instruct27k",
        REPO_ROOT / "datasets" / "PreVAD-Instruct27k" / "filter_test.json",
        "video_path",
    ),
    (
        "UCF-Crime",
        REPO_ROOT / "datasets" / "UCF-Crime" / "ucf-crime_test_anno.json",
        "video_path",
    ),
    (
        "XD-Violence",
        REPO_ROOT / "datasets" / "xd-violence" / "xd_test_anno.json",
        "video_path",
    ),
    (
        "MSAD",
        REPO_ROOT / "datasets" / "MSAD" / "msad_test_anno.json",
        "video_path",
    ),
]


def load_items(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def main():
    total_missing = 0

    for name, anno_path, key in CHECKS:
        if not anno_path.exists():
            print(f"[ERROR] Missing annotation file: {anno_path}")
            total_missing += 1
            continue

        items = load_items(anno_path)
        missing = []
        for item in items:
            raw_path = item.get(key)
            if not raw_path:
                missing.append("<empty video_path>")
                continue
            if not resolve_path(raw_path).exists():
                missing.append(raw_path)

        print(f"\n[{name}]")
        print(f"Annotation: {anno_path.relative_to(REPO_ROOT)}")
        print(f"Samples: {len(items)}")
        print(f"Missing videos: {len(missing)}")

        if missing:
            for sample in missing[:20]:
                print(f"  - {sample}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            total_missing += len(missing)

    if total_missing:
        raise SystemExit(1)

    print("\nAll referenced videos were found.")


if __name__ == "__main__":
    main()
