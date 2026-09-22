# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Predictors: callables record -> {"probabilities": {qid: {key: p}}, "latency_ms", "input_tokens", ...} for jevany.benchmark.

LocalPredictor scores a checkpoint in-process. RemotePredictor scores any compatible `/v1/systemone` endpoint.
"""
import json
import math
import time
import urllib.request

import torch

from jevany.api import question_keys
from jevany.checkpoint import Checkpoint, LoadOptions
from jevany.data import api_request, materialize
from jevany.device import sync
from jevany.model import MAX_PACKED


class ModelPredictor:
    """Adapt an in-memory DecisionModel to the benchmark predictor interface."""

    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device
        self.temperature = model.head.temperature

    @torch.no_grad()
    def __call__(self, record):
        enc = self.model.encode(self.tok, materialize(record), strict=True)
        if len(enc["ids"]) > MAX_PACKED:
            raise ValueError(f"packed request exceeds frozen {MAX_PACKED}-token limit")
        sync(self.device)
        start = time.perf_counter()
        logits = self.model.forward(enc)
        ps = [torch.softmax(z, -1).cpu() for z in logits]
        zs = [z.float().cpu() for z in logits]
        sync(self.device)
        keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in record["questions"].items()}
        return {"probabilities": {qid: dict(zip(keys[qid], p.tolist())) for qid, p in zip(keys, ps)},
                "logits": {qid: dict(zip(keys[qid], z.tolist())) for qid, z in zip(keys, zs)},
                "inference_temperature": self.temperature,
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": len(enc["ids"])}


class LocalPredictor(ModelPredictor):
    def __init__(self, run, device, opts=LoadOptions()):
        """opts.temperature=None scores with the temperature the checkpoint carries; 1.0 scores raw logits."""
        if opts.temperature is not None and not (math.isfinite(opts.temperature) and opts.temperature > 0):
            raise ValueError("temperature must be finite and positive")
        checkpoint = Checkpoint(run)
        self.run = checkpoint.path
        if device == "cuda":
            # evaluation is fp32-exact: TF32 (10-bit mantissa) moves probabilities by ~1e-3, the isolation gate's tolerance
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        tok, model = checkpoint.load(device, opts)
        super().__init__(model, tok, device)


class RemotePredictor:
    """Score any TypeSafe System One-compatible endpoint (POST <base_url>/v1/systemone) on frozen records. Probabilities are
    taken from the response as returned (renormalised by validate_distribution like every other predictor). Records the
    server-reported model id so the manifest can pin what was scored."""

    def __init__(self, base_url, model="jevany-27b", api_key="local", timeout=120, retries=3):
        self.base_url, self.model, self.api_key, self.timeout, self.retries = base_url.rstrip("/"), model, api_key, timeout, retries
        self.served_model = None

    def __call__(self, record):
        payload = json.dumps({**api_request(record), "model": self.model}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=payload, method="POST",
                                    headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(self.retries):
            try:
                start = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                latency = 1000 * (time.perf_counter() - start)
                break
            except Exception as error:   # 5xx / timeouts: retry with backoff; anything persistent surfaces as a rejected record
                last = error; time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"remote endpoint failed after {self.retries} attempts: {last}")
        self.served_model = body.get("model", self.served_model)
        probs = {}
        for qid, q in record["questions"].items():
            a = body["answers"][qid]
            if q["type"] == "noul": probs[qid] = {"true": float(a["noul"]), "false": 1 - float(a["noul"])}
            else: probs[qid] = {str(k): float(v) for k, v in a["probabilities"].items()}
        return {"probabilities": probs, "latency_ms": latency, "input_tokens": (body.get("usage") or {}).get("input_tokens")}
