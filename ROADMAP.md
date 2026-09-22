# Roadmap

JevAny starts with a narrow primitive: map a state and explicit options to a decision distribution. The next releases make the state richer, the tasks harder, and the decision process inspectable.

## 0.1 — Decision Core

- Release Qwen3.8-27B LoRA SFT and RLCR checkpoints.
- Keep training and serving on the same typed request format.
- Publish calibration, selective prediction, robustness, and latency results.
- Support multi-node DDP, distributed evaluation, checkpointing, and W&B.

Exit criterion: a clean public repository and reproducible local loading of both checkpoints.

## 0.2 — Complex and Agent Data

- Define multi-step decision tasks with observable intermediate state and verifiable terminal outcomes.
- Generate task candidates with language models, then deduplicate, validate, and retain source provenance.
- Add agent trajectories: tool selection, stop/continue, escalation, recovery, and risk-aware routing.
- Separate knowable, ambiguous, and missing-information cases instead of forcing every row to one hard answer.

Exit criterion: a frozen evaluation panel with no prompt or source overlap and human-audited subsets for every task family.

## 0.3 — Multimodal Decisions

- Connect the Qwen3.8 vision tower to the decision path.
- Support images, document pages, and sampled video clips in state.
- Test temporal order, cross-frame evidence, OCR, and absent-evidence calibration.

Exit criterion: image and video inputs share the same typed output contract and report modality-specific calibration.

## 0.4 — Long Context and Memory

- Replace the current 2,048-token training envelope with long-context training and evaluation.
- Add retrieval and state compression without hiding dropped evidence.
- Carry a bounded, inspectable memory across sequential decisions.

Exit criterion: accuracy and confidence remain stable as relevant evidence moves deeper into context.

## 0.5 — Test-Time Training and Harness

- Adapt on deployment feedback under an explicit budget.
- Add rollback, held-out canaries, contamination detection, and per-update provenance.
- Build a harness for task quality, calibration, option-order robustness, latency, cost, and downstream agent outcomes.

Exit criterion: every online update is reversible and must pass frozen safety and calibration gates before promotion.

## 0.6 — Symbolic Decision Programs

- Use an LLM to propose a task-specific decision tree from a schema, examples, and constraints.
- Validate each branch against labelled and counterfactual cases.
- Route tree leaves into JevAny questions so symbolic control flow and learned uncertainty remain separate.
- Record the tree version and branch path with every decision.

Exit criterion: generated trees are executable, auditable, and outperform a flat decision request on complex tasks without increasing confident errors.
