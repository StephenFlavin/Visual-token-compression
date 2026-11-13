"""Phase 2: small CRNN line readers trained from scratch, one per (scheme, em).

Modules
  data   - source lines, rendering + augmentation, pre-rendered uint8 caches, vocab, batching,
           the CTC frame-factor rule (T >= 2L+1 for every sample).
  model  - height-agnostic CNN -> BiLSTM -> linear (CTC).
  train  - training loop, greedy decoding, val/test CER, runs.jsonl rows.
  stubs  - synthetic schemes (multi-label cells, 3000-symbol alphabet) for tests/smoke only.
"""
