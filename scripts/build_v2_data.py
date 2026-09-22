#!/usr/bin/env python3
"""Build the diverse text, agent, image, and video mixture used for the next JevAny run."""
import argparse
import ast
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset

from jevany.data import materialize
from jevany.model import MAX_BRANCH, MAX_PACKED, MAX_STATE, fits, load_tokenizer
from jevany.suite import digest, read_jsonl, write_json, write_jsonl


SOURCES = {
    "helpsteer": ("nvidia/HelpSteer3", "preference", "f6d145777bcbde96137596340fab89793acd1031"),
    "hermes_single": ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"),
    "hermes_multi": ("NousResearch/hermes-function-calling-v1", "func_calling", "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"),
    "hermes_glaive": ("NousResearch/hermes-function-calling-v1", "glaive_func_calling", "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453"),
    "qasc": ("allenai/qasc", None, "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"),
    "commonsense": ("tau/commonsense_qa", None, "94630fe30dad47192a8546eb75f094926d47e155"),
    "scienceqa": ("derek-thomas/ScienceQA", None, "f18b0a70359ebfb41f658fd564208d0355b013f4"),
    "aokvqa": ("HuggingFaceM4/A-OKVQA", None, "d1b0efa3a436e9101dfbde3752db7607da696c35"),
    "mmmu": ("lmms-lab-encoder/MMMU", None, "364f2e2eb107b36e07ff4c5a15f5947a759cef47"),
    "ai2d": ("lmms-lab-encoder/ai2d", None, "c83a9b9692933aff8349157c88a413df9d02c4e5"),
    "video": ("TIGER-Lab/VideoFeedback", "real", "4557cb71595098e68790bb35253225c2fe797dcf"),
}
LICENSES = {
    "helpsteer": "CC-BY-4.0", "hermes_single": "Apache-2.0",
    "hermes_multi": "Apache-2.0", "hermes_glaive": "Apache-2.0", "arc": "CC-BY-SA-4.0",
    "qasc": "CC-BY-4.0", "commonsense": "MIT", "scienceqa": "CC-BY-SA-4.0",
    "video": "Apache-2.0", "aokvqa": "verify upstream terms before redistribution",
    "mmmu": "evaluation only; verify upstream terms", "ai2d": "evaluation only; verify upstream terms",
}
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
PREFERENCE_P2 = {-3: 0.02, -2: 0.10, -1: 0.30, 0: 0.50, 1: 0.70, 2: 0.90, 3: 0.98}


def load_source(name, split):
    repo, config, revision = SOURCES[name]
    return load_dataset(repo, config, split=split, revision=revision)


def stable_int(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "big")


def sample(rows, count, seed):
    if count <= 0:
        return []
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[: min(count, len(indices))]]


def content_hash(item):
    questions = {
        question_id: {key: value for key, value in question.items()
                      if key in ("type", "instructions", "criteria")}
        for question_id, question in item["questions"].items()
    }
    payload = json.dumps({"state": item["state"], "questions": questions}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(" ".join(payload.casefold().split()).encode()).hexdigest()


def metadata(source, identifier, split, state, questions, **extra):
    identifier = str(identifier)
    return {"source": source, "variant": "clean", "id": f"{source}/{identifier}",
            "group_id": f"{source}/{identifier}", "row": identifier, "split": split,
            "text_sha256": content_hash({"state": state, "questions": questions}), **extra}


def record(source, identifier, split, state, questions, media=None, **extra):
    result = {"state": state, "questions": questions}
    if media:
        result["media"] = media
    result["_meta"] = metadata(source, identifier, split, state, questions, **extra)
    return result


def helpsteer_record(row, index, split):
    score = int(row["overall_preference"])
    p2 = PREFERENCE_P2[score]
    questions = {"preferred_response": {
        "type": "choice",
        "instructions": "Which response is more helpful, correct, relevant, clear, and complete for this conversation?",
        "criteria": {"response_a": row["response1"], "response_b": row["response2"]},
        "label": "response_b" if score > 0 else "response_a",
        "target": {"response_a": round(1 - p2, 2), "response_b": p2},
        "src": "helpsteer3_preference",
    }}
    state = {"conversation": row["context"], "domain": row["domain"], "language": row["language"]}
    return record("helpsteer3", index, split, state, questions, preference_strength=abs(score))


def parse_tool_calls(text):
    calls = []
    for body in TOOL_CALL.findall(text):
        try:
            call = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            calls.append(call)
    return calls


def hermes_records(row, split, config="single"):
    try:
        tools = json.loads(row["tools"])
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(tools, list):
        return []
    conversations = row["conversations"]
    functions = [item.get("function", {}) for item in tools if item.get("type") == "function"]
    choices = {item.get("name"): item.get("description") or "No description provided" for item in functions if item.get("name")}
    choices["respond_without_tool"] = "Answer directly without calling a tool"
    if len(choices) < 2 or len(choices) > 64:
        return []
    output = []
    for answer_index, turn in enumerate(conversations):
        if turn["from"] not in ("gpt", "assistant"):
            continue
        calls = parse_tool_calls(turn["value"])
        selected = calls[0]["name"] if calls else "respond_without_tool"
        if selected not in choices:
            continue
        context = [{"role": item["from"], "content": item["value"]}
                   for item in conversations[:answer_index] if item["from"] != "system"]
        tool_summary = [{
            "name": function.get("name"),
            "description": function.get("description") or "No description provided",
            "arguments": list((function.get("parameters") or {}).get("properties", {})),
            "required_arguments": list((function.get("parameters") or {}).get("required", [])),
        } for function in functions]
        state = {"conversation": context, "available_tools": tool_summary}
        questions = {
            "next_action": {"type": "choice", "instructions": "What should the agent do next?",
                            "criteria": choices, "label": selected, "src": "agent_tool_selection"},
            "multiple_tools": {"type": "noul", "instructions": "Does this request require more than one tool call?",
                               "label": len(calls) > 1, "src": "agent_multi_tool"},
        }
        output.append(record(
            "agent_tool", f"{config}/{row['id']}/{answer_index}", split, state, questions,
            category=row.get("category"), agent_config=config,
            group_id=f"agent_tool/{config}/{row['id']}",
        ))
    return output


def choice_parts(row):
    choices = row.get("choices", row.get("options"))
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    if isinstance(choices, dict):
        texts, keys = choices["text"], choices.get("label") or [str(index) for index in range(len(choices["text"]))]
    else:
        texts, keys = choices, [str(index) for index in range(len(choices))]
    return [str(value) for value in keys], [str(value) for value in texts]


def answer_key(answer, keys, choices):
    answer = str(answer)
    if answer in keys:
        return answer
    if answer in choices:
        return keys[choices.index(answer)]
    if len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(answer.upper()) - ord("A")
        if index < len(keys):
            return keys[index]
    index = int(answer)
    return keys[index]


def text_choice_record(row, source, identifier, split):
    keys, choices = choice_parts(row)
    state = {key: row[key] for key in ("hint", "fact1", "fact2", "combinedfact") if row.get(key)}
    if not state:
        state = "Choose the best supported answer."
    question = row.get("question")
    if isinstance(question, dict):
        question = question.get("stem", str(question))
    questions = {"answer": {"type": "choice", "instructions": question, "criteria": dict(zip(keys, choices)),
                            "label": answer_key(row.get("answerKey", row.get("answer")), keys, choices),
                            "src": f"{source}_choice"}}
    return record(source, identifier, split, state, questions)


def save_image(image, path):
    if image is None:
        return False
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=90, optimize=True)
    return True


def visual_record(row, source, identifier, split, root):
    path = root / "media" / source / f"{identifier}.jpg"
    if not save_image(row.get("image"), path):
        return None
    if source == "aokvqa":
        choices = [str(value) for value in row["choices"]]
        answer = int(row["correct_choice_idx"])
        state = "Use the image and relevant world knowledge."
        question = row["question"]
    else:
        choices = [str(value) for value in row["choices"]]
        answer = int(row["answer"])
        state = {key: row[key] for key in ("hint", "subject", "topic", "skill") if row.get(key)}
        question = row["question"]
    keys = [str(index) for index in range(len(choices))]
    questions = {"answer": {"type": "choice", "instructions": question, "criteria": dict(zip(keys, choices)),
                            "label": str(answer), "src": f"{source}_vision"}}
    return record(source, identifier, split, state, questions,
                  media=[{"type": "image", "uri": str(path.relative_to(root))}])


def mmmu_record(row, split, root):
    if row.get("question_type") != "multiple-choice":
        return None
    images = []
    for index in range(1, 8):
        image = row.get(f"image_{index}")
        if image is None:
            continue
        path = root / "media" / "mmmu" / f"{row['id']}-{index}.jpg"
        save_image(image, path)
        images.append({"type": "image", "uri": str(path.relative_to(root))})
    if not images:
        return None
    keys, choices = choice_parts(row)
    questions = {"answer": {"type": "choice", "instructions": row["question"], "criteria": dict(zip(keys, choices)),
                            "label": answer_key(row["answer"], keys, choices), "src": "mmmu_vision"}}
    return record("mmmu", row["id"], split, {"subject": row.get("subfield", "")}, questions, media=images)


def ai2d_record(row, index, split, root):
    path = root / "media" / "ai2d" / f"{index}.jpg"
    if not save_image(row.get("image"), path):
        return None
    keys, choices = choice_parts(row)
    questions = {"answer": {"type": "choice", "instructions": row["question"], "criteria": dict(zip(keys, choices)),
                            "label": answer_key(row["answer"], keys, choices), "src": "ai2d_diagram"}}
    return record("ai2d", index, split, "Use the diagram to answer the question.", questions,
                  media=[{"type": "image", "uri": str(path.relative_to(root))}])


def download_video(row, split, root):
    path = root / "media" / "video_feedback" / f"{row['id']}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(row["video link"], temporary)
            temporary.replace(path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            print(f"video skipped {row['id']}: {error}", flush=True)
            return None
    levels = ["bad", "average", "good", "excellent"]
    dimensions = {
        "visual_quality": "visual quality", "temporal_consistency": "temporal consistency",
        "dynamic_degree": "dynamic degree", "text_alignment": "text-to-video alignment",
        "factual_consistency": "factual consistency",
    }
    records = []
    for name, description in dimensions.items():
        questions = {name: {
            "type": "score",
            "instructions": f"Rate the video's {description}.",
            "criteria": levels,
            "label": int(row[description]) - 1,
            "src": f"video_{name}",
        }}
        records.append(record(
            "video_feedback", f"{row['id']}/{name}", split,
            {"generation_prompt": row["text prompt"]}, questions,
            media=[{"type": "video", "uri": str(path.relative_to(root))}],
            group_id=f"video_feedback/{row['id']}",
        ))
    return records


def unique(records, excluded):
    seen, output = set(excluded), []
    for item in records:
        key = content_hash(item)
        if key not in seen:
            seen.add(key)
            item["_meta"]["text_sha256"] = key
            output.append(item)
    return output


def eval_split(rows, per_source, seed):
    by_source = {}
    for row in rows:
        by_source.setdefault(row["_meta"]["source"], []).append(row)
    calibration, development = [], []
    for source, group in sorted(by_source.items()):
        grouped = {}
        for row in group:
            grouped.setdefault(row["_meta"]["group_id"], []).append(row)
        bundles = list(grouped.values())
        random.Random(stable_int(f"{seed}:{source}")).shuffle(bundles)
        calibration_count = development_count = 0
        for bundle in bundles:
            if calibration_count < per_source:
                calibration.extend(bundle)
                calibration_count += len(bundle)
            elif development_count < per_source:
                development.extend(bundle)
                development_count += len(bundle)
            else:
                break
    return calibration, development


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--helpsteer", type=int, default=38000)
    parser.add_argument("--agent", type=int, default=12000)
    parser.add_argument("--hard-text", type=int, default=12000)
    parser.add_argument("--scienceqa", type=int, default=8000)
    parser.add_argument("--aokvqa", type=int, default=8000)
    parser.add_argument("--video", type=int, default=256)
    parser.add_argument("--eval-per-source", type=int, default=100)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include-train", action="append", default=[])
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--workers", type=int, default=16, help="parallel video downloads")
    parser.add_argument("--resume", action="store_true", help="reuse media already written by an interrupted build")
    args = parser.parse_args()
    root = Path(args.out)
    if root.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.cache:
        import datasets
        datasets.config.HF_DATASETS_CACHE = args.cache

    excluded = {content_hash(row) for path in args.exclude for row in read_jsonl(path)}
    train = [row for path in args.include_train for row in read_jsonl(path)]
    evaluation = []

    if args.helpsteer or args.eval_per_source:
        hs_train = load_source("helpsteer", "train")
        # Sample indices directly so record IDs remain stable across dataset releases.
        hs_indices = sample(list(range(len(hs_train))), args.helpsteer, args.seed)
        train.extend(helpsteer_record(hs_train[index], index, "train") for index in hs_indices)
        hs_validation = load_source("helpsteer", "validation")
        evaluation.extend(helpsteer_record(row, index, "evaluation") for index, row in enumerate(hs_validation)
                          if int(row["overall_preference"]) != 0)

    hermes_train, hermes_eval = [], []
    if args.agent or args.eval_per_source:
        for source in ("hermes_single", "hermes_multi", "hermes_glaive"):
            config = SOURCES[source][1]
            for row in load_source(source, "train"):
                row_key = f"{config}/{row['id']}"
                for converted in hermes_records(
                    row, "evaluation" if stable_int(row_key) % 20 == 0 else "train", config,
                ):
                    target = hermes_eval if converted["_meta"]["split"] == "evaluation" else hermes_train
                    target.append(converted)
    train.extend(sample(hermes_train, args.agent, args.seed + 1)); evaluation.extend(hermes_eval)

    text_sources = []
    if args.hard_text or args.eval_per_source:
        for name in ("arc", "qasc", "commonsense"):
            dataset = load_source(name, "train")
            text_sources.extend(text_choice_record(row, name, index, "train") for index, row in enumerate(dataset))
            validation = load_source(name, "validation")
            evaluation.extend(text_choice_record(row, name, index, "evaluation") for index, row in enumerate(validation))
    train.extend(sample(text_sources, args.hard_text, args.seed + 2))

    if args.scienceqa or args.eval_per_source:
        science_train = load_source("scienceqa", "train")
        if args.scienceqa:
            science_count = 0
            for index in sample(list(range(len(science_train))), len(science_train), args.seed + 3):
                converted = visual_record(science_train[index], "scienceqa", index, "train", root)
                if converted:
                    train.append(converted)
                    science_count += 1
                    if science_count >= args.scienceqa:
                        break
        science_validation = load_source("scienceqa", "validation")
        science_eval_count = 0
        for index in sample(list(range(len(science_validation))), len(science_validation), args.seed + 4):
            converted = visual_record(science_validation[index], "scienceqa", f"validation-{index}", "evaluation", root)
            if converted:
                evaluation.append(converted)
                science_eval_count += 1
                if science_eval_count >= args.eval_per_source * 3:
                    break

    if args.aokvqa or args.eval_per_source:
        aok_train = load_source("aokvqa", "train")
        for index in sample(list(range(len(aok_train))), args.aokvqa, args.seed + 5):
            row = aok_train[index]; converted = visual_record(row, "aokvqa", row["question_id"], "train", root)
            if converted: train.append(converted)
        aok_validation = load_source("aokvqa", "validation")
        for row in sample(aok_validation, args.eval_per_source * 2, args.seed + 6):
            converted = visual_record(row, "aokvqa", f"validation-{row['question_id']}", "evaluation", root)
            if converted: evaluation.append(converted)

    if args.eval_per_source:
        mmmu_count = 0
        for row in sample(load_source("mmmu", "validation"), 10 ** 9, args.seed + 7):
            converted = mmmu_record(row, "evaluation", root)
            if converted:
                evaluation.append(converted)
                mmmu_count += 1
                if mmmu_count >= args.eval_per_source * 3:
                    break
        for index, row in enumerate(sample(load_source("ai2d", "test"), args.eval_per_source * 2, args.seed + 8)):
            converted = ai2d_record(row, index, "evaluation", root)
            if converted: evaluation.append(converted)

    if args.video or args.eval_per_source:
        video_train = sample(load_source("video", "train"), args.video, args.seed + 9)
        video_evaluation = sample(load_source("video", "test"), min(80, args.eval_per_source * 2), args.seed + 10)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for converted in pool.map(lambda row: download_video(row, "train", root), video_train):
                train.extend(converted or [])
            for converted in pool.map(lambda row: download_video(row, "evaluation", root), video_evaluation):
                evaluation.extend(converted or [])

    train = unique(train, excluded)
    evaluation = unique(evaluation, excluded | {content_hash(row) for row in train})
    tokenizer = load_tokenizer(args.tokenizer, revision=args.tokenizer_revision or None)
    before = len(train)
    train = [row for row in train if fits(materialize(row), tokenizer)]
    before_evaluation = len(evaluation)
    evaluation = [row for row in evaluation if fits(materialize(row), tokenizer)]
    dropped = {"train": before - len(train), "evaluation": before_evaluation - len(evaluation)}
    calibration, development = eval_split(evaluation, args.eval_per_source, args.seed)
    random.Random(args.seed).shuffle(train)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "calibration.jsonl", calibration)
    write_jsonl(root / "development.jsonl", development)
    files = {name: {"records": len(rows), "questions": sum(len(row["questions"]) for row in rows),
                    "sha256": digest(root / name)} for name, rows in
             (("train.jsonl", train), ("calibration.jsonl", calibration), ("development.jsonl", development))}
    write_json(root / "manifest.json", {
        "name": "jevany-v2", "seed": args.seed, "files": files, "base_revisions": {},
        "trainable_sources": sorted({row["_meta"]["source"] for row in train}),
        "eval_only_sources": ["mmmu", "ai2d"],
        "holdout_sources": ["mmmu", "ai2d"], "sources": {
            key: {"repo": value[0], "config": value[1], "revision": value[2], "license": LICENSES[key]}
            for key, value in SOURCES.items()
        },
        "counts": {"train": dict(Counter(row["_meta"]["source"] for row in train)),
                   "calibration": dict(Counter(row["_meta"]["source"] for row in calibration)),
                   "development": dict(Counter(row["_meta"]["source"] for row in development))},
        "excluded_text_hashes": len(excluded), "context_dropped": dropped,
        "context": {"max_state": MAX_STATE, "max_branch": MAX_BRANCH, "max_packed": MAX_PACKED,
                    "truncate": False},
        "preference_target_probability": PREFERENCE_P2,
        "admission_tokenizer": {"name": args.tokenizer, "revision": args.tokenizer_revision or None},
        "included_training_files": {
            str(Path(path).resolve()): digest(path) for path in args.include_train
        },
    })
    print(json.dumps({"train": len(train), "calibration": len(calibration), "development": len(development),
                      "sources": Counter(row["_meta"]["source"] for row in train)}, indent=2))


if __name__ == "__main__":
    main()
