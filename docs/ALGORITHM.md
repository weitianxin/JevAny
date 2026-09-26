# Algorithm

## Decision Model

JevAny receives a shared state and one or more typed questions. Each question defines its candidate answers explicitly. It does not decode an answer string.

The encoder reuses existing Qwen delimiters where available. Other tokenizers
receive five decision tokens with trainable embeddings. Both use the same layout:

```text
<state> state
<question> instruction <option> a </option> ... <decide>
```

For recurrent backbones such as Qwen3.8, sliding-window backbones, and architectures
without validated packed-mask support, each question is a separate causal row
that repeats the state. This prevents one question from changing another
question's representation. Supported full-attention backbones can use the
equivalent packed block-causal mask.

Let `h_d` be the hidden state at `<decide>` and `h_i` the hidden state at the end of option `i`. The pointer head computes

```text
z_i = K(h_i) · Q(h_d) / sqrt(d_p)
p_i = softmax(z)_i
```

where `d_p=256` in the released checkpoints. Temperature scaling divides `z` by a scalar fitted on a separate calibration partition. It does not change the selected answer.

## Supervised Fine-Tuning

For a hard target `y`, SFT minimizes `-log p_y`. A row may instead carry a normalized soft target `t`, in which case the loss is `-Σ t_i log p_i`. Soft targets represent ambiguity or missing information without inventing a single correct label.

The released SFT adapter uses LoRA rank 16 over the Qwen attention, MLP, and Gated DeltaNet projections. The pointer head is trained with the adapter; the base weights remain frozen.

## RLCR

The paper *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty* introduces Reinforcement Learning with Calibration Rewards. Its reward augments correctness with a proper scoring rule:

```text
r(c, q) = c - (q - c)²
```

Here `c` is answer correctness and `q` is confidence in the selected answer. A confidently correct decision approaches `1`; a confidently wrong decision approaches `-1`.

JevAny adapts that idea to a pointer model:

1. Compute the option logits `z`.
2. Draw a group of 32 zero-mean Gaussian perturbations around `z`.
3. For each proposal, take its highest-probability option and that option's probability as confidence.
4. Compute the RLCR reward and subtract the group mean to form advantages.
5. Apply a score-function gradient to the proposal distribution.
6. Weight the policy and supervised cross-entropy terms explicitly to control decision drift.

For proposal `z'`, the location score uses the isotropic Gaussian log density

```text
log q(z' | z) = constant - Σᵢ (z'ᵢ - zᵢ)² / (2σ²)
```

The sum is over option dimensions. Averaging over options would make the policy gradient weaker as the choice set grows, which is especially harmful for the many-choice tasks in the RL mixture. Gradients are clipped to norm 1 after distributed synchronization. Training logs the total objective, the policy term, cross-entropy, reward, Brier penalty, and the gradient norm before clipping.

The selected experimental run decays exploration standard deviation from `0.4` to `0.2`, weights the policy term by `0.25`, and weights the supervised anchor by `0.5`.

This implementation borrows the calibration reward and group-relative baseline, but it is not the paper's generated-reasoning setup and is not standard token-level GRPO. There are no reasoning rollouts, confidence tokens, critic, or reference-model KL term. The policy is the distribution over perturbed pointer logits.

## Training Data Strategy

SFT is broad and deliberately diverse across preference, agent, visual, video, reasoning, classification, compositional, and policy tasks. RLCR is shorter and deliberately nonuniform:

- hard and many-choice reasoning;
- mathematical and medical decisions;
- agent and tool choices;
- preference, image-derived, and video-derived cases;
- broad replay to limit forgetting.

This makes RLCR a calibration and hard-case refinement stage, not a second full SFT pass. The current result improves development NLL slightly but does not improve overall transfer accuracy, so it remains experimental. See [DATA.md](DATA.md) for exact counts.
