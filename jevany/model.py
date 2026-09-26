# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Decision model: causal LM backbone + block-causal branch mask + pointer readout."""
import copy, math, os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import (
    LEGACY_TOKENS, decision_tokens, get_backbone_adapter, prepare_embeddings, prepare_tokenizer,
)

# Reuse existing rarely-used Qwen special tokens as delimiters (state, q, opt, /opt, decide) so no
# embedding rows need to be added/trained; LoRA adapts their meaning.
SPECIAL = LEGACY_TOKENS
# training context: state tokens, tokens per question branch, and the whole packed record. Frozen suites are admitted with
# this rule (jevany.suite) and training applies it to records built on the fly, so train and eval see the same population.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 1024, 2048, 2048


def load_tokenizer(name, revision=None):
    return get_backbone_adapter("text").load_preprocessor(name, revision)


def load_preprocessor(name, revision=None, multimodal=False, backbone_adapter="auto"):
    return get_backbone_adapter(backbone_adapter, multimodal=multimodal,
                                source=name, revision=revision).load_preprocessor(name, revision)


def tokenizer_of(preprocessor):
    return getattr(preprocessor, "tokenizer", preprocessor)


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def safe_text(text):
    return _SPECIAL_RE.sub(r"<¦\1¦>", text)


OPT_NONE, OPT_DECIDE = -1, -2   # values of enc["opt"]: instruction/state tokens, and the <decide> token


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, option_isolation=False):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after state),
    decide_idx [Q], opt_idx [Q][K] (index of </opt> token for each option), opt (per-token option index within its
    question: OPT_NONE for state/instruction, 0..K-1 for option spans, OPT_DECIDE for <decide>).

    option_isolation=True: every option span is its own sub-branch (it sees state + instruction + itself only), all
    option spans share the same position ids, and <decide> sits at one fixed position after the longest span. Then the
    per-option representations and <decide>'s attention over them are permutation-invariant by construction.
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ValueError(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    special = decision_tokens(tok)
    S = [tok.convert_tokens_to_ids(special[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in special[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ValueError(f"branch too long: {len(br)}")
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = list(range(p0, p0 + len(instr))) + [p0 + len(instr) + i for sp in spans for i in range(len(sp))] + [p0 + len(instr) + longest]
        else:
            br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        ids += br; seg += [k] * len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "option_isolation": option_isolation, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state_tokens) + 1 > max_state}


def media_to(value, device):
    """Move native processor tensors while retaining nested lists and metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: media_to(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(media_to(item, device) for item in value)
    return value


def encode_multimodal(processor, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, adapter=None):
    """Encode one image or video decision request with the base model's native multimodal processor."""
    media = rec.get("media") or []
    if not media:
        return encode(processor.tokenizer, rec, max_state=max_state, max_branch=max_branch, strict=strict)
    if len(rec["questions"]) != 1:
        raise ValueError("multimodal records must contain exactly one isolated question")
    adapter = adapter or get_backbone_adapter("qwen_vl")
    unsupported = {item["type"] for item in media} - adapter.media_types
    if unsupported:
        raise ValueError(f"{adapter.name} does not support media types: {sorted(unsupported)}")
    def text(value):
        value = safe_text(value)
        for token in processor.tokenizer.all_special_tokens:
            if token.startswith(("<", "[")):
                value = value.replace(token, token.replace("<", "‹").replace("[", "［"))
        return value
    special = decision_tokens(processor.tokenizer)
    parts = [special[0], text(rec["state"])]
    for question in rec["questions"]:
        parts.extend((special[1], text(question["instr"])))
        for option in question["options"]:
            parts.extend((special[2], text(option), special[3]))
        parts.append(special[4])
    batch = adapter.process_media(processor, media, "".join(parts))
    ids = batch["input_ids"][0].tolist()
    tok = processor.tokenizer
    q_id, close_id, decide_id = (tok.convert_tokens_to_ids(special[index]) for index in (1, 3, 4))
    question_starts = [index for index, token in enumerate(ids) if token == q_id]
    decide_idx = [index for index, token in enumerate(ids) if token == decide_id]
    if len(question_starts) != len(rec["questions"]) or len(decide_idx) != len(rec["questions"]):
        raise ValueError("multimodal delimiter layout mismatch")
    state_len = question_starts[0]
    text_state_len = len(user_tokens(tok, rec["state"])) + 1
    if strict and text_state_len > max_state:
        raise ValueError(f"multimodal text state exceeds {max_state} tokens: {text_state_len}")
    opt_idx, seg = [], [0] * state_len
    for index, (start, end, question) in enumerate(zip(question_starts, decide_idx, rec["questions"]), start=1):
        ends = [position for position in range(start, end) if ids[position] == close_id]
        if len(ends) != len(question["options"]):
            raise ValueError("multimodal option delimiter layout mismatch")
        if end - start + 1 + state_len > max_branch:
            raise ValueError(f"multimodal branch exceeds {max_branch} tokens")
        opt_idx.append(ends)
        seg.extend([index] * (end - start + 1))
    if len(seg) != len(ids):
        raise ValueError("multimodal branch layout mismatch")
    mm = media_to({key: value for key, value in batch.items() if key != "input_ids"}, "cpu")
    return {"ids": ids, "seg": seg, "pos": list(range(len(ids))), "opt": [OPT_NONE] * len(ids),
            "option_isolation": False, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]], "state_truncated": False,
            "multimodal": True, "mm": mm}


def fits(rec, *tokenizers, max_state=MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED):
    """True when the internal record encodes strictly (no truncation) within the training context under every tokenizer
    given (frozen suites are admitted against the tokenizers of all their pinned bases)."""
    try:
        return all(len(encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)["ids"]) <= max_packed for tok in tokenizers)
    except ValueError:
        return False


def branch_mask(seg, device, dtype=torch.float32):
    """attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]). Returns additive [1,1,L,L]."""
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(segs, device, dtype=torch.float32, opts=None, length=None):
    """Batched block-causal mask, additive [B,1,L,L], right-padded to the longest sequence.

    Padded key positions are masked for every query; padded query rows keep the diagonal so no row is fully
    masked (finfo.min, not -inf, so softmax stays finite either way). Real tokens never see pads because pads sit
    after them (causal) and belong to no segment (-1).

    opts (option isolation): within a question, an option-span token may attend to state, the instruction, and its own
    span only; <decide> attends to everything in its question. Instruction tokens never see option spans (causal)."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        o = torch.full((len(segs), L), OPT_NONE, device=device)
        for b, op in enumerate(opts):
            o[b, : len(op)] = torch.tensor(op, device=device)
        key_is_option = (o[:, None, :] >= 0)
        query_is_decide = (o[:, :, None] == OPT_DECIDE)
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_of(enc):
    """Split a packed encoding into its state and per-question branch rows.

    Returns (state_ids, state_pos, rows) with rows[k] = {"ids", "pos", "decide", "opts"}: the branch tokens of question
    k with their (already state-continuing) positions, and the readout offsets *within the branch*. Feeding
    state + rows[k] as one causal row is equivalent to the packed block-causal form for that question, on any
    architecture: the row contains exactly the tokens question k may attend to, in the same positions."""
    seg = enc["seg"]; Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1                                    # <decide> is the last token of its branch
        if seg[start] != k or seg[end - 1] != k: raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        """dp = pointer dimension (head capacity knob)."""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        # calibration: logits are divided by this at inference (eval mode) only. 1.0 = raw. A checkpoint carries the value fitted on
        # its in-distribution development rows (scripts/calibrate_checkpoint.py -> head.pt["temperature"]); training always sees T=1 so
        # a fitted value stays meaningful, and the argmax is unchanged by construction.
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False,
                 special_embeddings=False, lora_targets="all", dtype=torch.float32, multimodal=False,
                 backbone_adapter="auto", branch_mode="auto", lora_target_modules=""):
        super().__init__()
        tokenizer = tokenizer_of(tok)
        prepare_tokenizer(tokenizer)
        self.adapter = get_backbone_adapter(backbone_adapter, multimodal=multimodal, source=name, revision=revision)
        self.backbone_adapter = self.adapter.name if backbone_adapter == "auto" else backbone_adapter
        # backbone only (no vocab head): we never generate text.
        # eager on MPS/CPU (known-good with our float 4D mask); SDPA on CUDA (accepts arbitrary additive masks).
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        # Use fp32 for exact evaluation or bf16 to reduce accelerator memory.
        self.multimodal = multimodal
        self.lm, self.mm = self.adapter.load_model(name, revision=revision, dtype=dtype, attn=attn)
        added_token_ids = prepare_embeddings(self.lm, tokenizer)
        for module in self.adapter.frozen_modules(self.mm):
            module.requires_grad_(False)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # Recurrent, sliding-window and unrecognized backbones use independent
        # causal rows. Only adapters with validated mask support use packed input.
        cfg = self.lm.config
        self.hybrid = "linear_attention" in set(getattr(cfg, "layer_types", None) or [])
        if branch_mode not in ("auto", "packed", "rows"):
            raise ValueError("branch_mode must be auto, packed, or rows")
        packed = self.adapter.supports_packed(cfg)
        self.branch_mode = ("packed" if packed else "rows") if branch_mode == "auto" else branch_mode
        if self.branch_mode == "packed" and not packed:
            raise ValueError("this backbone adapter does not support packed masks; use branch_mode='rows'")
        if self.branch_mode == "rows" and option_isolation:
            raise ValueError("option_isolation requires a backbone with packed-mask support")
        self.option_isolation = option_isolation
        self.special_embeddings = bool(special_embeddings or added_token_ids)
        if lora:
            from peft import LoraConfig, get_peft_model
            embedding = self.lm.get_input_embeddings()
            embedding_name = next(name for name, module in self.lm.named_modules() if module is embedding)
            extra = {"trainable_token_indices": {embedding_name: [
                tokenizer.convert_tokens_to_ids(t) for t in decision_tokens(tokenizer)
            ]}} if self.special_embeddings else {}
            targets = self.adapter.lora_modules(self.lm, lora_targets, lora_target_modules)
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=targets, **extra)
            self.set_language_model(get_peft_model(self.lm, cfg))
        self.head = PointerHead(self.lm.get_input_embeddings().weight.shape[1], dp=head_dim)
        self.device = device
        self.to(device)

    def encode(self, tok, rec, **kw):
        """encode() with this model's option-isolation setting; use this from serving/eval code."""
        if rec.get("media"):
            if not self.multimodal:
                raise ValueError("checkpoint does not support media")
            if self.option_isolation:
                raise ValueError("option_isolation is not supported for native media records")
            return self.adapter.encode_media(tok, rec, **kw)
        return encode(tokenizer_of(tok), rec, option_isolation=self.option_isolation, **kw)

    def set_language_model(self, language_model):
        self.lm = language_model
        if self.mm is not None:
            self.adapter.attach_language_model(self.mm, language_model)

    def train(self, mode=True):
        super().train(mode)
        for module in self.adapter.frozen_modules(self.mm):
            module.eval()
        return self

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    SHAPE_BUCKET = int(os.environ.get("JEVANY_SHAPE_BUCKET", "64"))   # MPS: pad the sequence to a multiple of this (per-shape kernel warm-up); 1 disables

    def _pad_rows(self, rows):
        """Right-pad (ids, pos) token rows into [N, L] id / position tensors and a [N, L] attention mask (1 = real token).
        Pads sit after every real token and are masked keys, so they never change a real token's hidden state (parity
        measured exact). On MPS in eval mode L is rounded up to a SHAPE_BUCKET multiple so kernels are warmed per bucket."""
        L = max(len(ids) for ids, _ in rows)
        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device); pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device); att[i, : len(rid)] = 1
        return ids, pos, att

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch of encoded records under the packed block-causal mask."""
        ids, pos, _ = self._pad_rows([(e["ids"], e["pos"]) for e in encs])
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs):
            raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        lm_dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, dtype=lm_dtype, opts=[e["opt"] for e in encs] if isolate else None, length=ids.shape[1])
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()   # head stays fp32

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def forward_rows_batch(self, encs):
        """Row form: every question of every record is one causal row = state tokens + its branch tokens, right-padded
        into a single batch. Returns the same nested logits as forward_batch. Exact isolation by construction (rows are
        independent); the state is recomputed per row (Q x state tokens), which training accepts; serving uses the
        prefix cache instead."""
        rows, readouts = [], []   # one causal row per question; readouts[i] = (record, <decide> offset, option offsets)
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"])); readouts.append((b, len(S) + r["decide"], [len(S) + o for o in r["opts"]]))
        ids, pos, att = self._pad_rows(rows)
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att).last_hidden_state.float()
        out = [[] for _ in encs]
        for i, (b, d, oi) in enumerate(readouts):
            out[b].append(self.head(h[i, d], h[i, torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        """Returns list of logits tensors, one per question."""
        return self.forward_batch([enc])[0]

    def forward_multimodal(self, enc):
        if not self.multimodal:
            raise ValueError("multimodal encoding requires a multimodal checkpoint")
        kwargs = media_to(enc["mm"], self.device)
        ids = torch.tensor([enc["ids"]], device=self.device)
        kwargs.setdefault("attention_mask", torch.ones_like(ids))
        hidden = self.adapter.forward_media(self.lm, self.mm, {"input_ids": ids, **kwargs})[0].float()
        return self._readout(hidden, enc)

    def forward_batch(self, encs):
        """Per-record, per-question logits using the adapter's selected branch layout."""
        if any(enc.get("multimodal") for enc in encs):
            return [self.forward_multimodal(enc) if enc.get("multimodal") else self.forward_rows_batch([enc])[0] for enc in encs]
        if self.branch_mode == "rows": return self.forward_rows_batch(encs)
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    # --- state-prefix reuse (serving): the state is encoded once, question branches attend to its cached keys/values.
    # Exact by construction: branch tokens never attend to each other across questions (block-causal mask) and the state
    # never sees the branches (causal), so the state's hidden states and KV are identical with or without the branches.

    def _branch_rows_from_prefix(self, enc, cache):
        """Row-based serving: replicate the cached state once per question and run the branches as causal rows (exactly the
        forward_rows_batch layout, minus the recomputed state). Works on a copy: the caller's prefix stays pristine."""
        S, Sp, rows = rows_of(enc); Q = len(rows)
        cache = copy.deepcopy(cache); cache.reorder_cache(torch.zeros(Q, dtype=torch.long, device=self.device))
        ids, pos, att = self._pad_rows([(r["ids"], r["pos"]) for r in rows])
        att = torch.cat([torch.ones((Q, len(S)), dtype=torch.long, device=self.device), att], 1)   # the cached state tokens are all real
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, past_key_values=cache, use_cache=True).last_hidden_state.float()
        return [F.softmax(self.head(h[i, r["decide"]], h[i, torch.tensor(r["opts"], device=self.device)]), -1).cpu() for i, r in enumerate(rows)]

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens only. Returns (n_state_tokens, kv cache, state hidden states [Ls, d])."""
        if enc.get("multimodal"):
            raise ValueError("prefix caching does not support media; use probs()")
        Ls = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:Ls]], device=self.device); pos = torch.tensor([enc["pos"][:Ls]], device=self.device)
        # the cache must know the layer types (hybrid backbones keep recurrent + conv states per DeltaNet layer)
        out = self.lm(input_ids=ids, position_ids=pos, past_key_values=self.adapter.new_cache(self.lm), use_cache=True)
        return Ls, out.past_key_values, out.last_hidden_state[0].float()

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        """One full pass that also returns the state prefix (KV cropped to the state, state hidden states): a cache miss
        costs a single forward pass, not two."""
        if enc.get("multimodal"):
            raise ValueError("prefix caching does not support media; use probs()")
        Ls = enc["seg"].count(0)
        if self.branch_mode == "rows":
            # recurrent layers cannot be cropped back to the state, so a hybrid miss is a state pass (kept as the prefix)
            # plus the branch rows
            Ls, cache, h_state = self.prefix(enc)
            return self._branch_rows_from_prefix(enc, cache), (Ls, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device); pos = torch.tensor([enc["pos"]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)
        out = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, past_key_values=self.adapter.new_cache(self.lm), use_cache=True)
        h = out.last_hidden_state[0].float()
        out.past_key_values.crop(-(len(enc["ids"]) - Ls))     # keep the state only (negative = drop that many trailing tokens; positive form deprecated in transformers 5)
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)], (Ls, out.past_key_values, h[:Ls].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        """probs() for a record whose state tokens equal the cached prefix's; only the branches run. The cache is cropped
        back to the state afterwards so it can be reused."""
        if enc.get("multimodal"):
            raise ValueError("prefix caching does not support media; use probs()")
        Ls, cache, h_state = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        if self.branch_mode == "rows":
            return self._branch_rows_from_prefix(enc, cache)
        ids = torch.tensor([enc["ids"][Ls:]], device=self.device); pos = torch.tensor([enc["pos"][Ls:]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)[:, :, Ls:, :]
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=mask, use_cache=True)
            h = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - Ls))
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
