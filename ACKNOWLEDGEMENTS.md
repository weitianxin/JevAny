# Acknowledgements

JevAny includes modified infrastructure code from [Kev](https://github.com/jaredpalmer/kev), created by Jared Palmer and released under Apache-2.0. The required attribution is also recorded in `NOTICE` and in derived source files.

JevAny develops a separate research direction around calibration-aware reinforcement learning for decision models. Tianxin Wei created the current training system, RLCR implementation, checkpoints, evaluation, release tooling, and product roadmap.

The architecture was informed by the public analysis in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked) and the public System One API. JevAny does not contain Jev weights, outputs or private implementation details.

RLCR follows the reward proposed in *Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty* by Damani et al. JevAny adapts it to a prefill-only pointer model and does not reproduce the paper's generated reasoning rollouts.

Qwen3.8-27B is produced by the Qwen team. The adapters in this project require its separately distributed base weights.

The Phi-4 vision conversion helper adapts configuration and weight-name mappings from Hugging Face Transformers under Apache-2.0. It merges Microsoft's pretrained vision LoRA before JevAny training.

The README teaser uses Lucide icons. The [source records](docs/icons/sources.json) and [license notices](docs/icons/LICENSE) accompany the editable SVG.
