# Tasks

## BEiT-3 Experiment A

- [x] Complete 10 epochs and full VQA v2 public validation on two Tesla T4 GPUs.
- [x] Match all 214,354 validation IDs and repeat the official-toolkit evaluation twice; confirm 74.22 and preserve the full epoch history and curves.
- [x] Close A at the verified result. Document the 0.78-point gap to the original research target.
- [x] Recommend the verified BEiT-3 epoch-9 checkpoint and retain public ViLT downloads as a fallback.
- [x] Build the `beit3-a-v1` inference ZIP, verify all 497 FP32 tensors bit-for-bit and pass offline CPU inference.
- [x] Upload the BEiT-3 ZIP, checksum and upstream notice; match each GitHub asset size and SHA-256 to the local export.
- [ ] Optional follow-up: run the predeclared B comparison and C independent-seed replication before reassessing the promotion criteria.

## VQA Studio

- [x] Include SwiftUI sources, local inference worker, artwork, build and validation scripts, and desktop-focused tests. Keep the generated app bundle and personal data out of Git.
- [ ] Separate release task: evaluate signing and notarization for standalone distribution.
