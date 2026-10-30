### What this PR does / why we need it?

Fixes #{ISSUE_NUMBER}

In PD disaggregation with decode TP < prefill TP, `KVCacheRecvingThread._stash_pending_reformat` assigns the whole per-`(request_id, shard_idx)` bucket on every call:

```python
with self.pending_reformat_lock:
    self.pending_reformat[request_id][shard_idx] = ready_attention_group_reformat_block_ids
```

Attention groups of the same shard become transfer-complete in different pull tasks — a 1-KV-head group (MLA / indexer) is `is_group_transfer_end` on its first pull, a multi-KV-head GQA group only on its last (`_get_attention_group_num_need_pulls_for_decode_tp` is computed per group) — and those tasks run on different `ThreadPoolExecutor` threads, each stashing once. The later stash drops the groups stashed earlier, so `_reformat_pending_kv_caches` reorganizes only the last-arriving group and the dropped group's KV is consumed in the un-reformatted cross-rank layout. In a GLM-family + DSpark stack this silently corrupts the speculative draft's KV once blocks are recycled, collapsing acceptance length from 4+ to ~1.9 while output correctness (target-model KV) is unaffected. Full analysis, dose–response data, and reproduction signature: #{ISSUE_NUMBER}.

This PR merges into the bucket instead of overwriting it:

| Aspect | Before | After |
|---|---|---|
| Groups stashed by an earlier call | dropped on the next call | kept |
| Duplicate arrival of a group already stashed | n/a (whole bucket replaced) | skipped (dedup by `group_idx`) |
| Shard ready in a single batch (decode TP ≥ prefill TP, or single attention group) | correct | byte-identical behavior |
| Lock scope / consumer `_reformat_pending_kv_caches` | — | unchanged |

Changes:

| File | Change |
|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | `_stash_pending_reformat`: merge + dedup into the existing per-shard bucket instead of assignment |
| `tests/ut/distributed/kv_transfer/kv_p2p/test_mooncake_connector.py` | new regression test `test_stash_pending_reformat_merges_batches_arriving_separately` |
| `release_notes/fragments/<PR number>.yaml` | release note fragment (bugfix), pushed as a follow-up commit once the PR number is assigned |

### Does this PR introduce _any_ user-facing change?

No API or configuration change. Behavioral bugfix: PD disaggregation with decode TP < prefill TP no longer corrupts the KV of multi-KV-head (GQA) transfer groups; speculative-decoding acceptance length recovers accordingly. A release note fragment is added in this PR as a follow-up commit once the PR number is assigned.

### How was this patch tested?

- New unit test `test_stash_pending_reformat_merges_batches_arriving_separately` in `tests/ut/distributed/kv_transfer/kv_p2p/test_mooncake_connector.py` (same `KVCacheRecvingThread.__new__` style as the existing tests in that file). It fails on the current overwrite implementation and asserts: single-batch equivalence with the old behavior, merge of batches arriving separately, dedup of a repeated group, shard independence, and that the consumer drains every stashed group exactly once.
- Standalone differential harness over the real function bodies (`_stash_pending_reformat` + `_reformat_pending_kv_caches` extracted from the patched file): 4000 randomized stash/consume sequences; single-batch scenarios byte-identical to the original semantics, multi-batch scenarios match the merge-with-dedup oracle.
- End-to-end on two independent dual-node Ascend NPU PD clusters (GLM-family target + DSpark GQA draft, 64k inputs, temperature 0): before the fix, recycled-block requests collapse to acceptance length ~1.9 (deterministic, bit-identical across days); after the fix, 16/16 probes healthy, recycled-block acceptance ≥ 3.3, and the fresh-vs-recycled two-state signature disappears. Decode TP8 (no reformat path) and single-node serving are unaffected before and after.
- `pre-commit` hooks on the changed files pass (ruff check / ruff format / codespell / typos / repo-local checks).

---

### Checklist

#### Testing

- [x] **Boundary / edge-case coverage**: I reviewed whether this change involves
  boundary conditions (e.g. empty input, single/zero element, min/max values,
  off-by-one, sequence-length limits, integer overflow). Where applicable, I
  added unit tests covering these edge cases in `tests/ut/` or `tests/e2e/`.
- [x] **Regression test**: If this PR fixes a bug, I added a regression test
  that would fail without the fix.
- [ ] **Error / failure path coverage**: I covered abnormal paths (exceptions,
  timeouts, out-of-range, division by zero) in addition to the happy path.
- [x] **Meaningful & reliable tests**: New tests assert real behavior and are
  not written merely to satisfy CI.
- [ ] **Performance impact**: If this PR touches a performance-sensitive path,
  I attached benchmark results and added performance regression tests.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01RuXuePUhAJ8pzCBeWXaqEg
