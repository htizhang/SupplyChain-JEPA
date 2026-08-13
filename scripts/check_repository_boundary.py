"""Fail closed when release-forbidden files or environment metadata enter Git."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TEXT_SCAN_EXCLUSIONS = {Path("uv.lock")}
BINARY_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".joblib",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
# Assemble deployment-provider names so this policy file does not flag itself.
FORBIDDEN_TEXT = (
    "a" + "ws",
    "s3" + "://",
    "ec" + "2",
    "sage" + "maker",
    "ama" + "zon",
    "/" + "Users/",
    "/" + "home/",
    "/" + "opt/ml",
    "v3" + ".3",
    "research" + "-workspace",
)


def repository_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def check_repository() -> list[str]:
    failures: list[str] = []
    for path in repository_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in BINARY_SUFFIXES:
            failures.append(f"forbidden binary or dataset: {relative}")
        if path.is_file() and path.stat().st_size > 2 * 1024 * 1024:
            failures.append(f"file exceeds 2 MiB release limit: {relative}")
        if path == SELF or relative in TEXT_SCAN_EXCLUSIONS or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            failures.append(f"unreviewed binary file: {relative}")
            continue
        for marker in FORBIDDEN_TEXT:
            if marker.lower() in text:
                failures.append(f"environment-specific marker {marker!r}: {relative}")
    return failures


def main() -> int:
    failures = check_repository()
    if failures:
        print("\n".join(failures))
        return 1
    print(f"repository boundary passed for {len(repository_files())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
