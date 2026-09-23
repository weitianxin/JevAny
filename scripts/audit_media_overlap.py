#!/usr/bin/env python3
"""Audit exact and perceptual media overlap between an evaluation suite and training media."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from jevany.suite import digest, read_json, read_jsonl, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def phash(image):
    values = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low_frequency = dctn(values, norm="ortho")[:8, :8]
    median = np.median(low_frequency.flat[1:])
    bits = low_frequency > median
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def image_signature(path):
    with Image.open(path) as image:
        return {
            "hashes": [phash(image)],
            "width": image.width,
            "height": image.height,
            "frames": 1,
            "bytes": path.stat().st_size,
        }


def video_signature(path):
    import av
    hashes = []
    with av.open(str(path)) as container:
        streams = container.streams.video
        if not streams:
            raise ValueError(f"no video stream: {path}")
        stream = streams[0]
        duration = container.duration
        if not duration:
            raise ValueError(f"video duration unavailable: {path}")
        for fraction in (0.2, 0.5, 0.8):
            container.seek(int(duration * fraction), backward=True)
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise ValueError(f"cannot decode sampled frame: {path}")
            hashes.append(phash(frame.to_image()))
        return {
            "hashes": hashes,
            "width": stream.width,
            "height": stream.height,
            "frames": stream.frames or None,
            "duration_seconds": float(duration / 1_000_000),
            "bytes": path.stat().st_size,
        }


def signatures(paths, kind, workers):
    function = image_signature if kind == "image" else video_signature
    results, errors = {}, {}

    def inspect(path):
        try:
            return path, function(path), None
        except Exception as error:
            return path, None, f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, value, error in pool.map(inspect, paths):
            if error:
                errors[str(path)] = error
            else:
                results[path] = value
    return results, errors


def near_overlap(evaluation, reference, reference_splits, kind, threshold):
    matches = []
    required = 1 if kind == "image" else 3
    for eval_path, eval_value in evaluation.items():
        best = None
        for reference_path, reference_value in reference.items():
            if len(eval_value["hashes"]) != len(reference_value["hashes"]):
                continue
            distances = [
                (left ^ right).bit_count()
                for left, right in zip(eval_value["hashes"], reference_value["hashes"])
            ]
            score = max(distances) if required > 1 else distances[0]
            if best is None or score < best[0]:
                best = (score, reference_path, distances)
        if best and best[0] <= threshold:
            matches.append({
                "evaluation": str(eval_path),
                "reference": str(best[1]),
                "reference_splits": sorted(reference_splits[best[1]]),
                "distances": best[2],
            })
    return matches


def referenced_media(suite, splits, kind):
    suite = Path(suite)
    paths = {}
    for split in splits:
        path = suite / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            for item in row.get("media", []):
                if item["type"] == kind:
                    media_path = (suite / item["uri"]).resolve()
                    if not media_path.is_file():
                        raise FileNotFoundError(media_path)
                    paths.setdefault(media_path, set()).add(split)
    return paths


def normalize_text(value):
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, dict):
        return {str(key): normalize_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_text(item) for item in value]
    return value


def question_fingerprint(question):
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        values = list(criteria.values())
    elif isinstance(criteria, list):
        values = criteria
    else:
        values = []
    options = sorted(
        json.dumps(normalize_text(value), sort_keys=True, separators=(",", ":"))
        for value in values
    )
    payload = {
        "instructions": normalize_text(question.get("instructions")),
        "options": options,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def question_index(rows, split):
    hashes, ids = {}, set()
    for row in rows:
        identifier = row["_meta"]["id"]
        ids.add(identifier)
        for question_id, question in row["questions"].items():
            hashes.setdefault(question_fingerprint(question), []).append({
                "id": identifier, "question": question_id, "split": split,
            })
    return hashes, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-suite", help="suite whose JSONL media references define prior exposure")
    reference.add_argument("--reference-media", help="directory scan for cases without a reference suite")
    parser.add_argument("--reference-splits", default="train,calibration,development")
    parser.add_argument("--kind", choices=("image", "video"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--record-in-manifest", action="store_true")
    args = parser.parse_args()
    suite = Path(args.suite)
    suffixes = IMAGE_SUFFIXES if args.kind == "image" else VIDEO_SUFFIXES
    rows = read_jsonl(suite / "development.jsonl")
    eval_paths = sorted({
        (suite / item["uri"]).resolve()
        for row in rows for item in row.get("media", [])
        if item["type"] == args.kind
    })
    if args.reference_suite:
        split_names = [value.strip() for value in args.reference_splits.split(",") if value.strip()]
        reference_membership = referenced_media(args.reference_suite, split_names, args.kind)
        reference_root = Path(args.reference_suite)
        reference_policy = {
            "suite": str(reference_root),
            "manifest_sha256": digest(reference_root / "manifest.json"),
            "splits": {
                split: digest(reference_root / f"{split}.jsonl")
                for split in split_names
            },
        }
        reference_questions, reference_ids = {}, set()
        for split in split_names:
            split_hashes, split_ids = question_index(
                read_jsonl(Path(args.reference_suite) / f"{split}.jsonl"), split,
            )
            reference_ids.update(split_ids)
            for value, items in split_hashes.items():
                reference_questions.setdefault(value, []).extend(items)
    else:
        reference_membership = {
            path.resolve(): {"directory"}
            for path in Path(args.reference_media).rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        }
        reference_policy = {"media_directory": str(Path(args.reference_media))}
        reference_questions, reference_ids = {}, set()
    reference_paths = sorted(reference_membership)
    eval_signatures, eval_errors = signatures(eval_paths, args.kind, args.workers)
    reference_signatures, reference_errors = signatures(reference_paths, args.kind, args.workers)
    exact = set(map(digest, eval_paths)) & set(map(digest, reference_paths))
    near = near_overlap(
        eval_signatures, reference_signatures, reference_membership,
        args.kind, args.threshold,
    )
    match_counts = {
        split: sum(split in match["reference_splits"] for match in near)
        for split in sorted({value for values in reference_membership.values() for value in values})
    }
    eval_questions, eval_ids = question_index(rows, "evaluation")
    question_matches = [
        {"evaluation": item, "reference": reference_questions[value]}
        for value, items in eval_questions.items() if value in reference_questions
        for item in items
    ]
    result = {
        "kind": args.kind,
        "reference": reference_policy,
        "perceptual_hash": "64-bit DCT pHash",
        "near_duplicate_threshold": args.threshold,
        "video_rule": "all three sampled-frame hashes must be within the threshold" if args.kind == "video" else None,
        "evaluation_files": len(eval_paths),
        "reference_files": len(reference_paths),
        "evaluation_decode_errors": eval_errors,
        "reference_decode_errors": reference_errors,
        "exact_sha256_overlap": len(exact),
        "near_duplicate_matches": near,
        "near_duplicate_matches_by_reference_split": match_counts,
        "question_option_fingerprint": "sha256 of normalized instructions and sorted option values; option keys and order ignored",
        "question_option_matches": question_matches,
        "source_id_overlap": sorted(eval_ids & reference_ids),
        "evaluation": {
            "bytes": {
                "minimum": min(value["bytes"] for value in eval_signatures.values()),
                "maximum": max(value["bytes"] for value in eval_signatures.values()),
            },
            "width": {
                "minimum": min(value["width"] for value in eval_signatures.values()),
                "maximum": max(value["width"] for value in eval_signatures.values()),
            },
            "height": {
                "minimum": min(value["height"] for value in eval_signatures.values()),
                "maximum": max(value["height"] for value in eval_signatures.values()),
            },
        },
    }
    if args.kind == "video":
        result["evaluation"]["duration_seconds"] = {
            "minimum": min(value["duration_seconds"] for value in eval_signatures.values()),
            "maximum": max(value["duration_seconds"] for value in eval_signatures.values()),
        }
    write_json(args.out, result)
    if args.record_in_manifest:
        manifest_path = suite / "manifest.json"
        output_path = Path(args.out).resolve()
        if output_path.parent != suite.resolve():
            raise ValueError("--record-in-manifest requires the audit output inside the suite")
        manifest = read_json(manifest_path)
        manifest.setdefault("artifacts", {})[output_path.name] = {"sha256": digest(output_path)}
        media_audit = manifest.setdefault("media_audit", {})
        for obsolete in ("training_files_compared", "training_unique_sha256", "exact_training_overlap"):
            media_audit.pop(obsolete, None)
        media_audit.update({
            "reference": reference_policy,
            "reference_files_compared": len(reference_paths),
            "exact_reference_overlap": len(exact),
            "perceptual_reference_overlap": len(near),
            "evaluation_decode_errors": len(eval_errors),
            "reference_decode_errors": len(reference_errors),
        })
        write_json(manifest_path, manifest)
    print(f"audited {len(eval_paths)} evaluation and {len(reference_paths)} referenced {args.kind} files")
    print(f"exact overlap={len(exact)}, near matches={len(near)}, eval decode errors={len(eval_errors)}")


if __name__ == "__main__":
    main()
