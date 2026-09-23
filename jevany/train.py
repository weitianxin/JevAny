# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Fine-tune the decision model on labelled requests (a frozen suite's training partition, records built on the
fly from the public sources, or your own JSONL), with the pointer head trained from scratch.

    uv run python -m jevany.train --suite data/decision-v7 --out runs/sft
    uv run python -m jevany.train --data data/rlcr.jsonl --init_from runs/sft --rlcr --out runs/rlcr

Batch size is small (variable-length records with custom masks) and gradients are accumulated over --accum micro-batches.
"""
import argparse, contextlib, json, math, os, random, resource, sys, time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from .checkpoint import Checkpoint, Meta, write_meta
from .device import allocated_bytes, default_device, empty_cache
from .data import EVAL_ONLY, augment, load_records, materialize, none_pair, source_seed
from .suite import ENCODING, digest, load_split, read_manifest, validate_training, write_json
from .model import MAX_BRANCH, MAX_PACKED, MAX_STATE, DecisionModel, load_preprocessor


# --- losses -----------------------------------------------------------------------------------------------------------

def question_loss(z, q, dev):
    """Cross-entropy against a hard label or a supplied soft target."""
    if q.get("target") is not None:
        t = torch.tensor(q["target"], device=dev, dtype=z.dtype)
        return -(t * F.log_softmax(z, -1)).sum()
    y = torch.tensor([q["label"]], device=dev)
    return F.cross_entropy(z[None], y)


def rlcr_reward(correctness, confidence):
    """The bounded correctness-plus-Brier reward from Damani et al.: c - (q - c)^2."""
    return correctness - (confidence - correctness).square()


def gaussian_location_log_probability(proposals, location, sigma):
    """Log density up to constants for isotropic Gaussian pointer proposals."""
    return -((proposals - location.unsqueeze(0)).square().sum(-1) / (2 * sigma ** 2))


def rlcr_question_loss(z, q, dev, group_size, sigma, ce_weight):
    """Group-relative policy gradient over noisy pointer logits, with the selected option probability as confidence."""
    if q.get("target") is not None:
        target = torch.tensor(q["target"], device=dev, dtype=z.dtype)
    else:
        target = F.one_hot(torch.tensor(q["label"], device=dev), len(z)).to(z.dtype)
    noise = torch.randn((group_size, len(z)), device=dev, dtype=z.dtype) * sigma
    noise = noise - noise.mean(-1, keepdim=True)
    proposals = z.detach().unsqueeze(0) + noise
    probabilities = F.softmax(proposals, -1)
    actions = probabilities.argmax(-1)
    confidence = probabilities.gather(-1, actions[:, None]).squeeze(-1)
    correctness = target[actions]
    reward = rlcr_reward(correctness, confidence)
    advantage = reward - reward.mean()
    # Isotropic Gaussian location log probability. Sum over option dimensions;
    # averaging here would suppress the policy gradient as the choice count grows.
    log_probability = gaussian_location_log_probability(proposals, z, sigma)
    policy_loss = -(advantage.detach() * log_probability).mean()
    ce = question_loss(z, q, dev)
    return (policy_loss + ce_weight * ce, ce, policy_loss, reward.mean(),
            (confidence - correctness).square().mean(), correctness.mean())


def accumulation_records(n, batch, accum, microbatch):
    start = (microbatch // accum) * accum * batch
    return min(accum * batch, n - start)


def evaluation_due(step, total_steps, interval, before_start=False):
    """Whether to evaluate at this optimizer step. The final step is always included when interval evaluation is on."""
    if step == 0:
        return before_start
    return interval > 0 and (step % interval == 0 or step == total_steps)


def limit_complete_groups(records, limit):
    """Take the groups represented by the first `limit` rows, including every later sibling in those groups."""
    if not limit or len(records) <= limit:
        return records
    groups = {record["_meta"]["group_id"] for record in records[:limit]}
    return [record for record in records if record["_meta"]["group_id"] in groups]


def distributed_slice(records, rank, world_size):
    """Return one equal-sized DDP shard, padding like DistributedSampler when needed."""
    if world_size == 1:
        return records, 0
    per_rank = math.ceil(len(records) / world_size)
    padding = per_rank * world_size - len(records)
    padded = records + records[:padding]
    return padded[rank::world_size], padding


def distributed_group_slice(records, rank, world_size):
    """Shard evaluation records without separating contrastive or permutation siblings."""
    slots = {}
    for record in records:
        group = record["_meta"]["group_id"]
        if group not in slots:
            slots[group] = len(slots) % world_size
    return [record for record in records if slots[record["_meta"]["group_id"]] == rank]


def reduce_counter(values, device):
    """Sum the per-rank training counters; a no-op outside a process group."""
    keys = ("objective", "ce", "rlcr_policy", "rlcr_reward", "rlcr_brier", "rlcr_correct",
            "rlcr_n", "grad_norm", "n", "tokens")
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return Counter(dict(zip(keys, tensor.cpu().tolist())))


def reduce_max(value, device):
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


# --- data -------------------------------------------------------------------------------------------------------------

def training_requests(a, tok, manifest, model):
    """Load labelled JSONL or a frozen suite, optionally mixing suite replay into custom data."""
    # the suite's rules (declared trainable sources, no held-out structures) apply to every record taken from it
    if a.data:
        reqs = load_records(a.data)
        if a.replay:
            pool = load_split(a.suite, "train"); replay = random.Random(f"replay:{a.seed}").sample(pool, min(a.replay, len(pool)))
            validate_training(replay, manifest)
            print(f"replay: {len(replay)} of {len(pool)} suite training records mixed with {len(reqs)} from {a.data}", flush=True)
            reqs = reqs + replay
    elif manifest:
        reqs = load_split(a.suite, "train"); validate_training(reqs, manifest)
    else:
        raise ValueError("pass --data or --suite")
    if not manifest or a.data:
        # Custom records have not passed suite admission, so apply the same context rule before strict encoding.
        kept = []
        for request in reqs:
            try:
                encoding = model.encode(tok, materialize(request), strict=True)
                if len(encoding["ids"]) <= MAX_PACKED:
                    kept.append(request)
            except ValueError:
                pass
        if len(kept) < len(reqs):
            print(f"dropped {len(reqs) - len(kept)} of {len(reqs)} records that exceed the training context "
                  f"({MAX_STATE} state / {MAX_BRANCH} branch / {MAX_PACKED} packed tokens)", flush=True)
        reqs = kept
    if not reqs:
        raise ValueError("empty training set")
    eval_only = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", []) if manifest else [])
    forbidden = {r["_meta"]["source"] for r in reqs} & eval_only
    if forbidden:
        raise ValueError(f"training data contains eval-only sources: {sorted(forbidden)}")
    return reqs


@dataclass
class Variant:
    """One augmented and encoded training example."""
    rec: dict
    enc: dict

    @property
    def tokens(self):
        return len(self.enc["ids"])


def encode_batch(model, tok, a, chunk, epoch):
    """Apply the release recipe's augmentations and encode each request."""
    out = []
    for req in chunk:
        item_rng = random.Random(source_seed(a.seed, f"{epoch}:{req['_meta']['id']}"))
        variants = [augment(req, item_rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract)]
        if a.p_none_pair > 0 and item_rng.random() < a.p_none_pair:
            variants += none_pair(req, item_rng)
        for v in variants:
            rec = materialize(v)
            enc = model.encode(tok, rec, strict=True)
            if len(enc["ids"]) > MAX_PACKED:
                raise ValueError(f"training request exceeds {MAX_PACKED} packed tokens")
            out.append(Variant(rec, enc))
    return out


class DistributedBatchForward(torch.nn.Module):
    """Expose the model's batched forward through DDP."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encodings):
        return self.model.forward_batch(encodings)


def batch_loss(model, a, batch, dev, autocast, distributed_forward=None, rlcr_sigma=None):
    """Return the summed SFT or RLCR objective and its logging terms."""
    terms = Counter()
    with autocast:
        encodings = [v.enc for v in batch]
        if distributed_forward is None:
            logits_b = model.forward_batch(encodings)
        else:
            logits_b = distributed_forward(encodings)
    loss = 0.0
    for v, logits in zip(batch, logits_b):
        if a.rlcr:
            scored = [rlcr_question_loss(z.float(), q, dev, a.rlcr_group_size, rlcr_sigma, a.rlcr_ce_w)
                      for z, q in zip(logits, v.rec["questions"])]
            objective = sum(item[0] for item in scored) / len(scored)
            ce = sum(item[1] for item in scored) / len(scored)
            policy = sum(item[2] for item in scored) / len(scored)
            terms["objective"] += objective.item()
            terms["rlcr_policy"] += policy.item()
            terms["rlcr_reward"] += sum(item[3].item() for item in scored) / len(scored)
            terms["rlcr_brier"] += sum(item[4].item() for item in scored) / len(scored)
            terms["rlcr_correct"] += sum(item[5].item() for item in scored) / len(scored)
            terms["rlcr_n"] += 1
            terms["ce"] += ce.item(); loss = loss + objective
        else:
            ce = sum(question_loss(z.float(), q, dev)
                     for z, q in zip(logits, v.rec["questions"])) / len(logits)
            terms["objective"] += ce.item(); terms["ce"] += ce.item(); loss = loss + ce
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    return loss, terms


# --- run --------------------------------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=0.0, help="separate learning rate for the pointer head (0 = same as --lr); the head trains from scratch")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay on LoRA and head parameters")
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--rlcr", action="store_true", help="optimize correctness plus Brier reward over noisy pointer distributions")
    ap.add_argument("--rlcr_group_size", type=int, default=32, help="noisy answer-confidence candidates per question")
    ap.add_argument("--rlcr_sigma_start", type=float, default=0.4, help="initial standard deviation of pointer-logit exploration")
    ap.add_argument("--rlcr_sigma_end", type=float, default=0.1, help="final standard deviation of pointer-logit exploration")
    ap.add_argument("--rlcr_ce_w", type=float, default=0.25, help="supervised CE retained beside the RLCR policy loss")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--suite", help="frozen suite directory; train only on its training partition")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--batch", type=int, default=1, help="records per forward pass (padded batch); optimizer step every --accum micro-batches")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = autocast forward with fp32 master weights (CUDA only)")
    ap.add_argument("--weights_dtype", choices=["fp32", "bf16"], default="fp32", help="dtype of the frozen backbone weights. bf16 halves memory and is required by the fused MoE experts "
                                                                                        "(torch._grouped_mm wants bf16); LoRA and head stay fp32 (peft upcasts adapters). The checkpoint records it and is loaded the same way.")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=0)
    ap.add_argument("--option_isolation", type=int, choices=[0, 1], default=0, help="option spans are isolated sub-branches with shared positions (exact permutation invariance)")
    ap.add_argument("--special_embeddings", type=int, choices=[0, 1], default=0, help="also train the embeddings of the 5 delimiter tokens")
    ap.add_argument("--multimodal", action="store_true", help="load the Qwen vision tower and accept image or video media entries")
    ap.add_argument("--head_dim", type=int, default=256, help="pointer head dimension")
    ap.add_argument("--lora_targets", choices=["all", "dense", "attn", "qv"], default="all", help="LoRA module set; fewer modules = less drift from the base; dense = all minus the DeltaNet projections on hybrid bases")
    ap.add_argument("--base_revision", default="", help="pin the base commit when the suite manifest does not pin this base")
    ap.add_argument("--p_none", type=float, default=0.1)
    ap.add_argument("--p_none_distract", type=float, default=0.12)
    ap.add_argument("--p_distract", type=float, default=0.15)
    ap.add_argument("--p_none_pair", type=float, default=0.0, help="fraction of Choice records that additionally emit a none-present/none-absent minimal pair")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--data", default="", help="your own labelled requests, one JSON object per line (see jevany.data.load_records); an alternative to --suite for fine-tuning, or combined with --suite and --replay")
    ap.add_argument("--replay", type=int, default=0, help="with --data and --suite: mix in this many records sampled (by --seed) from the suite's training partition, so a delta fine-tune does not forget the released recipe")
    ap.add_argument("--init_from", default="", help="delta mode: warm-start LoRA and the pointer head from an existing run "
                                                   "(local directory or hub id) instead of starting from the base model; keeps the "
                                                   "released model's in-domain skill while adapting to a new domain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps; 0 runs the full schedule")
    ap.add_argument("--eval_before_start", action="store_true", help="score the evaluation suite before the first optimizer step")
    ap.add_argument("--eval_every_steps", type=int, default=0, help="score after every N optimizer steps and at the final step; 0 disables periodic evaluation")
    ap.add_argument("--eval_records", type=int, default=0, help="deterministic record limit per evaluation split; 0 uses the full split")
    ap.add_argument("--eval_suite", default="", help="suite to score during training; defaults to --suite")
    ap.add_argument("--eval_transfer_suite", default="", help="optional out-of-domain suite whose development split is scored with the fitted temperature")
    ap.add_argument("--checkpoint_every_steps", type=int, default=0, help="save a model checkpoint after evaluations at this step interval; 0 saves only the final model")
    ap.add_argument("--wandb_project", default="", help="log training and periodic evaluation metrics to this W&B project")
    ap.add_argument("--wandb_name", default="")
    ap.add_argument("--wandb_group", default="")
    ap.add_argument("--wandb_entity", default="")
    ap.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    a = ap.parse_args()
    if min(a.epochs, a.accum, a.lora, a.batch) < 1:
        ap.error("epochs, accum, lora and batch must be positive")
    if a.dtype == "bf16" and a.device != "cuda":
        ap.error("--dtype bf16 requires --device cuda")
    if a.lr <= 0 or a.head_lr < 0 or a.weight_decay < 0:
        ap.error("invalid learning rate or weight decay")
    if a.replay and not (a.data and a.suite):
        ap.error("--replay needs both --data and --suite")
    if min(a.max_steps, a.eval_every_steps, a.eval_records, a.checkpoint_every_steps) < 0:
        ap.error("--max_steps, --eval_every_steps, --eval_records and --checkpoint_every_steps must be nonnegative")
    if (a.eval_before_start or a.eval_every_steps) and not (a.eval_suite or a.suite):
        ap.error("training-time evaluation needs --eval_suite or --suite")
    if a.checkpoint_every_steps and not a.eval_every_steps:
        ap.error("--checkpoint_every_steps requires --eval_every_steps")
    if a.rlcr_group_size < 2 or min(a.rlcr_sigma_start, a.rlcr_sigma_end) <= 0 or a.rlcr_ce_w < 0:
        ap.error("RLCR needs group_size >= 2, positive sigmas and nonnegative CE weight")
    if Path(a.out).exists() and int(os.environ.get("RANK", "0")) == 0:
        ap.error("refusing to overwrite an existing run")
    return a


def start_wandb(a, out_dir):
    if not a.wandb_project or a.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("--wandb_project requires the wandb package") from error
    run = wandb.init(project=a.wandb_project, entity=a.wandb_entity or None, group=a.wandb_group or None,
                     name=a.wandb_name or Path(a.out).name, mode=a.wandb_mode, dir=str(out_dir), config=vars(a))
    run.define_metric("optimizer_step")
    run.define_metric("train/*", step_metric="optimizer_step")
    run.define_metric("eval/*", step_metric="optimizer_step")
    return run


def distributed_evaluate_records(records, predictor, evaluation, name, temperature, heldout_sources):
    """Run model inference on group-preserving rank shards, then let rank 0 write the canonical benchmark report."""
    from .benchmark import evaluate_records

    rank, world_size = dist.get_rank(), dist.get_world_size()
    shard_dir = Path(evaluation) / "_distributed" / name
    if rank == 0:
        shard_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    shard = distributed_group_slice(records, rank, world_size)
    with (shard_dir / f"rank-{rank:03d}.jsonl").open("w", encoding=ENCODING, newline="\n") as output:
        for record in shard:
            output.write(json.dumps({"id": record["_meta"]["id"], "prediction": predictor(record)}, allow_nan=False) + "\n")
    dist.barrier()
    report = rows = None
    if rank == 0:
        predictions = {}
        for path in sorted(shard_dir.glob("rank-*.jsonl")):
            with path.open(encoding=ENCODING) as source:
                for line in source:
                    item = json.loads(line)
                    if item["id"] in predictions:
                        raise ValueError(f"duplicate distributed evaluation record: {item['id']}")
                    predictions[item["id"]] = item["prediction"]
        expected = {record["_meta"]["id"] for record in records}
        if predictions.keys() != expected:
            raise ValueError(f"distributed evaluation coverage mismatch: {len(predictions)}/{len(expected)}")

        class CachedPredictor:
            def __init__(self, values, inference_temperature):
                self.values, self.temperature = values, inference_temperature

            def __call__(self, record):
                return self.values[record["_meta"]["id"]]

        report, rows = evaluate_records(records, CachedPredictor(predictions, predictor.temperature),
                                        Path(evaluation) / name, temperature, heldout_sources)
    dist.barrier()
    return report, rows


def evaluate_during_training(model, tok, dev, suite, out_dir, step, record_limit=0, transfer_suite=None):
    """Score calibration and development splits in place, then restore the model's prior train/eval state."""
    from .benchmark import evaluate_records
    from .metrics import fit_temperature
    from .predictors import ModelPredictor

    evaluation = Path(out_dir) / "training_eval" / f"step-{step:06d}"
    records = {split: load_split(suite, split) for split in ("calibration", "development")}
    if record_limit:
        records = {split: limit_complete_groups(rows, record_limit) for split, rows in records.items()}
    was_training = model.training
    cuda = str(dev).startswith("cuda")
    if cuda:
        cuda_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                      torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    model.eval()
    predictor = ModelPredictor(model, tok, dev)
    try:
        if dist.is_initialized():
            _, calibration_rows = distributed_evaluate_records(records["calibration"], predictor, evaluation, "calibration", 1.0, ())
            values = [fit_temperature(calibration_rows, aggregation="micro") if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(values, src=0)
            temperature = values[0]
            report, _ = distributed_evaluate_records(records["development"], predictor, evaluation, "development", temperature,
                                                       tuple(read_manifest(suite).get("holdout_sources", [])))
        else:
            _, calibration_rows = evaluate_records(records["calibration"], predictor, evaluation / "calibration")
            temperature = fit_temperature(calibration_rows, aggregation="micro")
            report, _ = evaluate_records(records["development"], predictor, evaluation / "development", temperature,
                                         heldout_sources=tuple(read_manifest(suite).get("holdout_sources", [])))
        transfer = None
        if transfer_suite:
            transfer_records = load_split(transfer_suite, "development")
            if record_limit: transfer_records = limit_complete_groups(transfer_records, record_limit)
            heldout = tuple(read_manifest(transfer_suite).get("holdout_sources", []))
            if dist.is_initialized():
                transfer, _ = distributed_evaluate_records(transfer_records, predictor, evaluation, "transfer", temperature, heldout)
            else:
                transfer, _ = evaluate_records(transfer_records, predictor, evaluation / "transfer", temperature,
                                               heldout_sources=heldout)
    finally:
        model.train(was_training)
        if cuda:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = cuda_flags[:2]
            torch.backends.cuda.enable_flash_sdp(cuda_flags[2]); torch.backends.cuda.enable_mem_efficient_sdp(cuda_flags[3])
    summary = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        summary = {"optimizer_step": step, "temperature": temperature, "record_limit": record_limit,
                   "objective": report["objective"], "clean": report["clean"], "calibrated_clean": report["calibrated_clean"],
                   "coverage": report["coverage"], "transfer": transfer}
        write_json(evaluation / "summary.json", summary)
        history = Path(out_dir) / "training_eval" / "history.jsonl"
        with history.open("a", encoding=ENCODING, newline="\n") as output:
            output.write(json.dumps(summary, allow_nan=False) + "\n")
    if dist.is_initialized():
        dist.barrier()
    return summary


def wandb_eval_metrics(summary):
    metrics = {"optimizer_step": summary["optimizer_step"], "eval/objective": summary["objective"],
               "eval/temperature": summary["temperature"]}
    for prefix in ("clean", "calibrated_clean"):
        for name in ("acc", "nll", "ece", "brier", "confident_error_rate", "coverage_at_5pct_error"):
            metrics[f"eval/{prefix}/{name}"] = summary[prefix][name]
    if summary.get("transfer"):
        for prefix in ("clean", "calibrated_clean"):
            for name in ("acc", "nll", "ece", "brier", "confident_error_rate", "coverage_at_5pct_error"):
                metrics[f"eval/transfer/{prefix}/{name}"] = summary["transfer"][prefix][name]
    return metrics


def pinned_revision(a, manifest):
    """The base commit this run trains against: the suite's pin, or --base_revision when the suite has none."""
    revision = manifest["base_revisions"].get(a.base) if manifest else None
    if a.base_revision:
        if revision and revision != a.base_revision: raise ValueError("--base_revision conflicts with the suite's pinned revision")
        revision = a.base_revision
    if manifest and not revision:
        raise ValueError("base not pinned by the suite; pass --base_revision")
    return revision


def main():
    a = parse_args()
    dev = a.device or default_device()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if dev != "cuda":
            raise ValueError("distributed training requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank = 0
    main_process = rank == 0
    out_dir = Path(a.out)
    if main_process:
        out_dir.mkdir(parents=True)
    if distributed:
        dist.barrier()
    tracker = start_wandb(a, out_dir) if main_process else None
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else contextlib.nullcontext()
    manifest = read_manifest(a.suite) if a.suite else None
    revision = pinned_revision(a, manifest)
    holdout = manifest["holdout_sources"] if manifest else []
    tok = load_preprocessor(a.base, revision=revision, multimodal=a.multimodal)
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision,
                          head_dim=a.head_dim, lora_targets=a.lora_targets,
                          option_isolation=bool(a.option_isolation), special_embeddings=bool(a.special_embeddings),
                          dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32,
                          multimodal=a.multimodal)
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    # what this run will save as head.pt; also the architecture a warm start must match
    meta = Meta(base=a.base, base_revision=revision, lora=a.lora, head_dim=a.head_dim,
                option_isolation=bool(a.option_isolation),
                special_embeddings=bool(a.special_embeddings), multimodal=a.multimodal,
                weights_dtype=a.weights_dtype, holdout=holdout)
    init_source = None
    if a.init_from:
        # Start from an already trained adapter and pointer head for the RLCR or domain-adaptation stage.
        init_source = Checkpoint(a.init_from).warm_start(model, meta)
        if main_process:
            print(f"delta: warm start from {init_source['resolved']}: {init_source['adapter_tensors']} adapter tensors and the pointer head loaded", flush=True)
    if main_process:
        print(f"device={dev} world_size={world_size} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)

    reqs = training_requests(a, tok, manifest, model)
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    _, padding_per_epoch = distributed_slice(reqs, rank, world_size)
    per_rank_records = math.ceil(len(reqs) / world_size)
    micro_per_epoch = math.ceil(per_rank_records / a.batch)
    planned_steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    steps = min(planned_steps, a.max_steps) if a.max_steps else planned_steps
    global_batch = a.batch * a.accum * world_size
    if main_process:
        write_json(out_dir / "training_config.json", {"args": vars(a), "suite_sha256": suite_hash, "base_revision": revision,
                   "init_source": init_source, "holdout": holdout,
                   "distributed": {"world_size": world_size, "global_effective_batch": global_batch,
                                   "per_rank_records": per_rank_records, "padding_records_per_epoch": padding_per_epoch}})
        print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
              f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")
        print(f"world_size={world_size} local_batch={a.batch} accum={a.accum} global_effective_batch={global_batch} "
              f"optimizer_steps={steps}/{planned_steps}", flush=True)
        if tracker:
            tracker.config.update({"world_size": world_size, "global_effective_batch": global_batch,
                                   "planned_optimizer_steps": planned_steps})

    head_params = list(model.head.parameters()); head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.1)
    distributed_forward = None
    if distributed:
        device_index = torch.cuda.current_device()
        # The base is immutable and pinned, so synchronize the 22M trainable parameters without broadcasting the 70 GB
        # frozen backbone. DDP synchronizes their gradients from the first forward onward.
        for parameter in model.trainable_parameters():
            dist.broadcast(parameter.data, src=0)
        distributed_forward = DistributedDataParallel(DistributedBatchForward(model), device_ids=[device_index],
                                                       output_device=device_index, broadcast_buffers=False, init_sync=False)
        torch.manual_seed(a.seed + rank)
    eval_suite = a.eval_suite or a.suite
    model.train(); t0 = time.time(); run = Counter(); step_run = Counter(); step = seen = tokens_seen = peak_mem = 0
    last_eval_step = None

    def save_checkpoint(directory, checkpoint_step):
        directory = Path(directory)
        if main_process:
            directory.mkdir(parents=True, exist_ok=directory == out_dir)
        if distributed:
            dist.barrier()
        if main_process:
            model.lm.save_pretrained(directory)
        if main_process:
            checkpoint_meta = replace(meta, head=model.head.state_dict(),
                                      extra={"args": vars(a), "suite_sha256": suite_hash, "init_source": init_source,
                                             "optimizer_step": checkpoint_step})
            write_meta(directory, checkpoint_meta)
            tok.save_pretrained(directory)
        if distributed:
            dist.barrier()

    def run_evaluation():
        nonlocal last_eval_step
        summary = evaluate_during_training(model, tok, dev, eval_suite, out_dir, step, a.eval_records,
                                           a.eval_transfer_suite or None)
        save_due = step > 0 and a.checkpoint_every_steps and (step % a.checkpoint_every_steps == 0 or step == steps)
        if main_process:
            print(f"eval step {step}/{steps} acc {summary['clean']['acc']:.3f} nll {summary['clean']['nll']:.3f} "
                  f"cal_nll {summary['calibrated_clean']['nll']:.3f} T {summary['temperature']:.3f}", flush=True)
            if tracker: tracker.log(wandb_eval_metrics(summary))
        if save_due:
            checkpoint = out_dir / "checkpoints" / f"step-{step:06d}"
            save_checkpoint(checkpoint, step)
            if main_process:
                print(f"checkpoint step {step}: {checkpoint}", flush=True)
        if distributed:
            dist.barrier()
        last_eval_step = step

    if evaluation_due(0, steps, a.eval_every_steps, a.eval_before_start):
        run_evaluation()
    for ep in range(a.epochs):
        rng.shuffle(reqs)
        epoch_reqs, _ = distributed_slice(reqs, rank, world_size)
        for mb in range(micro_per_epoch):
            chunk = epoch_reqs[mb * a.batch : (mb + 1) * a.batch]
            batch = encode_batch(model, tok, a, chunk, ep)
            sync_now = (mb + 1) % a.accum == 0 or mb + 1 == micro_per_epoch
            sync_module = distributed_forward
            sync_context = sync_module.no_sync() if sync_module is not None and not sync_now else contextlib.nullcontext()
            progress = step / max(steps - 1, 1)
            rlcr_sigma = a.rlcr_sigma_start + progress * (a.rlcr_sigma_end - a.rlcr_sigma_start)
            with sync_context:
                loss, terms = batch_loss(model, a, batch, dev, autocast, distributed_forward,
                                         rlcr_sigma=rlcr_sigma)
                # Weight by source records in the local accumulation group. DDP averages these equal-sized rank means.
                group_records = accumulation_records(len(epoch_reqs), a.batch, a.accum, mb) * (len(batch) / len(chunk))
                (loss / group_records).backward()
            step_run += terms; step_run["n"] += len(batch); step_run["tokens"] += sum(v.tokens for v in batch)
            peak_mem = max(peak_mem, allocated_bytes(dev))
            if sync_now:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                step_run["grad_norm"] = float(grad_norm)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if dev == "mps": empty_cache(dev)   # MPS only: per-step cache release keeps the unified-memory footprint down; on CUDA it would just slow the step
                reduced = reduce_counter(step_run, dev)
                run += reduced; seen += int(reduced["n"]); tokens_seen += int(reduced["tokens"])
                if tracker:
                    tracker.log({"optimizer_step": step, "train/epoch": ep,
                                 "train/objective": reduced["objective"] / reduced["n"],
                                 "train/ce": reduced["ce"] / reduced["n"],
                                 "train/rlcr_policy": reduced["rlcr_policy"] / max(reduced["rlcr_n"], 1),
                                 "train/rlcr_reward": reduced["rlcr_reward"] / max(reduced["rlcr_n"], 1),
                                 "train/rlcr_brier": reduced["rlcr_brier"] / max(reduced["rlcr_n"], 1),
                                 "train/rlcr_correct": reduced["rlcr_correct"] / max(reduced["rlcr_n"], 1),
                                 "train/rlcr_sigma": rlcr_sigma,
                                 "train/grad_norm_pre_clip": reduced["grad_norm"] / world_size,
                                 "train/lr": sched.get_last_lr()[0], "train/records_seen": seen,
                                 "train/forward_tokens": tokens_seen, "train/seconds_per_record": (time.time() - t0) / seen})
                step_run = Counter()
                if step % 10 == 0 and main_process:
                    print(f"ep{ep} step {step}/{steps} objective {run['objective']/run['n']:.3f} "
                          f"ce {run['ce']/run['n']:.3f} policy {run['rlcr_policy']/max(run['rlcr_n'],1):.3f} "
                          f"reward {run['rlcr_reward']/max(run['rlcr_n'],1):.3f} "
                          f"grad_norm {reduced['grad_norm']/world_size:.3f} {(time.time()-t0)/seen:.3f}s/rec", flush=True)
                    run = Counter()
                if evaluation_due(step, steps, a.eval_every_steps):
                    run_evaluation()
                if step >= steps:
                    break
        if step >= steps:
            break

    if a.eval_every_steps and last_eval_step != step:
        run_evaluation()

    peak_mem = reduce_max(peak_mem, dev)
    peak_rss = reduce_max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024), dev)
    save_checkpoint(out_dir, step)
    if main_process:
        write_json(out_dir / "training_metrics.json", {"wall_seconds": time.time() - t0, "records_seen": seen,
                   "requested_records": a.epochs * len(reqs), "distributed_padding_records": a.epochs * padding_per_epoch,
                   "truncated_records": 0, "rejected_records": 0, "optimizer_steps": step,
                   "planned_optimizer_steps": planned_steps, "forward_tokens": tokens_seen,
                   "peak_device_bytes": peak_mem, "device": dev, "dtype": a.dtype, "batch": a.batch,
                   "world_size": world_size, "global_effective_batch": global_batch, "peak_rss_bytes": peak_rss})
        if tracker:
            write_json(out_dir / "wandb.json", {"id": tracker.id, "name": tracker.name, "project": tracker.project,
                                                 "entity": tracker.entity, "url": tracker.url})
            tracker.finish()
        print("saved", a.out, flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
