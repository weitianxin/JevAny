# Acknowledgements

JevAny began as a focused derivative of [Kev](https://github.com/jaredpalmer/kev), created by Jared Palmer and released under Apache-2.0. It retains the typed request format, pointer readout, branch isolation, calibration metrics and checkpoint layout from Kev.

This release changes the scope and implementation around Qwen3.8-27B. Tianxin Wei added distributed 16-GPU training and evaluation, the Qwen3.8 SFT recipe, decision-only RLCR, new checkpoints, a reduced package, new release tooling and the JevAny documentation.

The architecture was informed by the public analysis in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked) and the public System One API. JevAny does not contain Jev weights, outputs or private implementation details.

RLCR follows the reward proposed in *Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty* by Damani et al. JevAny adapts it to a prefill-only pointer model and does not reproduce the paper's generated reasoning rollouts.

Qwen3.8-27B is produced by the Qwen team. The adapters in this project require its separately distributed base weights.
