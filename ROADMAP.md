# Roadmap

JevAny starts with one primitive: map shared state and explicit options to a probability distribution. Each project below keeps that contract while changing the evidence, control flow, or learning loop.

## Now

### Jev-Judge

- [x] Release a 27B LoRA SFT decision model.
- [x] Train and evaluate a calibration-reward RL continuation.
- [x] Support choice, binary, and ordinal outputs through one API.
- [x] Report calibration, selective prediction, robustness, and transfer metrics.
- [ ] Improve RL without losing development or transfer accuracy.

### Jev-Agent And Jev-Tool

- [x] Evaluate legal-action selection on RAGEN FrozenLake and Sokoban.
- [x] Record complete action traces and render animated episodes.
- [x] Compile tool selection and escalation questions from an external LLM.
- [ ] Add coding, browser, computer-use, and robot-control environments.
- [ ] Train on recovery, stop, rollback, and human-escalation decisions.

### Jev-Harness And Jev-Symbolic

- [x] Connect a Bedrock planner to the System One decision API.
- [x] Hide evidence values from the planner by default.
- [x] Validate generated trees for cycles, missing branches, and unreachable nodes or outcomes.
- [x] Record the selected outcome and branch trace.
- [ ] Compare generated trees with flat requests on a frozen complex-task panel.
- [ ] Add cost, latency, calibration, and downstream outcome gates.

### Jev-Test

- [x] Run a locked no-ground-truth adaptation study on MMLU-Pro and MuSR.
- [x] Compare pseudo-label SFT and calibration-reward RL.
- [x] Publish the negative result without tuning on post-adaptation gold scores.
- [ ] Test softer vote distributions, confidence filters, canaries, and automatic rollback.

## Next

### Jev-Image And Jev-Video

- [x] Connect native images and sampled video frames to the decision path.
- [x] Train on A-OKVQA and ScienceQA images; exercise the native VideoFeedback path.
- [x] Evaluate native AI2D and MMMU images.
- [x] Evaluate a label-balanced MVBench panel with blank and group-shuffled video controls.
- [x] Evaluate a decontaminated MMStar panel with blank and group-shuffled image controls.
- [ ] Add broader document-page, temporal-ordering, OCR, and absent-evidence tests.
- [ ] Report modality-specific calibration and failure slices.

### Long Context

- [ ] Train beyond the current 2,048-token envelope.
- [ ] Measure evidence-position sensitivity and distractor robustness.
- [ ] Add retrieval and state compression with explicit evidence accounting.
- [ ] Carry bounded, inspectable memory across sequential decisions.

## Later

- [ ] Distill task-specific symbolic trees from successful trajectories.
- [ ] Let an LLM propose trees while JevAny verifies every internal decision.
- [ ] Support online adaptation only when frozen safety and calibration gates pass.
- [ ] Publish reproducible suites for coding, computer use, embodied control, and long-horizon agents.

Each release must keep data provenance, checkpoint hashes, frozen evaluation inputs, and reversible update history.
