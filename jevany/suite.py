# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Read immutable JSONL suites and verify their manifests."""
import hashlib
import json
from pathlib import Path

from .data import EVAL_ONLY

ENCODING = "utf-8"
SPLITS = ("train", "calibration", "development", "test")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record_digest(record):
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding=ENCODING))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding=ENCODING)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding=ENCODING).splitlines() if line.strip()]


def write_jsonl(path, records):
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    Path(path).write_text(body, encoding=ENCODING)


def read_manifest(directory):
    return read_json(Path(directory) / "manifest.json")


def load_split(directory, split, allow_test=False):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit --allow-test")
    directory = Path(directory)
    manifest = read_manifest(directory)
    path = directory / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing suite partition: {path}")
    expected = manifest["files"][path.name]
    if digest(path) != expected["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = read_jsonl(path)
    if len(records) != expected["records"]:
        raise ValueError(f"suite record count mismatch: {path}")
    return records


def validate_training(records, manifest):
    allowed = set(manifest.get("trainable_sources", []))
    forbidden = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) | set(manifest.get("holdout_sources", []))
    for record in records:
        source = record["_meta"]["source"]
        if source in forbidden or (allowed and source not in allowed):
            raise ValueError(f"eval-only or undeclared training source: {source}")
    if not records:
        raise ValueError("empty training partition")
