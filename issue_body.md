### Summary

In PD disaggregation with the mooncake p2p connector, when the decode side runs a **smaller TP size than the prefill side**, `KVCacheRecvingThread._stash_pending_reformat` **overwrites** the per-`(request_id, shard_idx)` stash of pending KV reformat work every time it is called:

https://github.com/vllm-project/vllm-ascend/blob/0bbf9248abdab3fa4256dc955bdb57abc06229e8/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1091-L1098

Attention groups of the **same shard** become transfer-complete in **different pull tasks** (details below), so the stash is called more than once per shard, and the later call silently drops the groups stashed earlier. The dropped groups are never reorganized by `_reformat_pending_kv_caches`, and their KV is consumed by the decode side in the un-reformatted (cross-rank interleaved) layout.

In practice the dropped group is always a **multi-KV-head (GQA) transfer group** — in a GLM-family + DSpark stack, the speculative draft model's KV. The target model's output is never affected (its MLA / indexer groups have 1 KV head and skip reformatting), so the corruption is **invisible to output-correctness monitoring** and shows up only as a collapsed speculative-decoding acceptance length.

The defective assignment is present on `main` (`0bbf9248`), `releases/v0.26.0rc`, `releases/v0.30.0`, and `releases/v0.31.0rc`, and was reproduced end-to-end on both the 0.26 and 0.30 stacks. It was introduced with the stash itself in #12359 (commit `1db626510`), i.e. the bug exists since the batched-reformat mechanism was added.

### Your current environment

Reproduced on two independent dual-node Ascend NPU PD clusters (one prefill node + one decode node, mooncake p2p connector); a full `collect_env.py` dump cannot be attached from those clusters, but the bug is structural in the connector code and the trigger condition is fully described below.

- vllm-ascend: reproduced on the v0.26.0 and v0.30.0 lines; defective code verified unchanged on `main` @ `0bbf9248`, `releases/v0.26.0rc`, `releases/v0.30.0`, `releases/v0.31.0rc`
- Serving form: PD disaggregation, 1P1D across two nodes, prefill TP8; decode TP varied as the experiment variable (8 / 2 / 1)
- Models: GLM-family target (MLA attention + SFA indexer caches) + DSpark speculative draft (3 layers, GQA, **8 KV heads**, sliding window 2048), k=7 drafts
- Workload: 64k-token inputs, `temperature=0`

Acceptance length below is defined as `1 + Σ accepted / Σ drafting steps`, i.e. the mean number of tokens committed per decode step including the anchor token.

### Symptom

**Two-state behavior keyed to KV block reuse.** For every freshly started decode engine, the *first* request (whose KV lands in freshly zero-initialized blocks) is healthy; every *subsequent* request (whose KV lands in recycled blocks) collapses:

| Request | KV blocks written | Acceptance length |
|---|---|---|
| first after engine start | freshly zero-initialized | 4.13 – 4.71 |
| all subsequent | recycled | 1.88 – 1.97 |

Observed on 8/8 engine launches with no counterexample. For a fixed prompt the two readings reproduce **bit-identically across days** (e.g. `4.1280` / `1.9033`), ruling out random noise: this is deterministic data corruption whose visible magnitude depends only on what was previously in the blocks (zero-filled blocks ≈ mild noise; recycled blocks = contamination by a previous request's data).

**Sustained load.** 64k inputs at concurrency 12: mean acceptance length **1.89**.

**Dose–response on decode TP** (only variable changed; prefill TP8):

| decode TP | pulls for the 8-KV-head draft group | Result |
|---|---|---|
| TP8 | 1 (reformat path not taken) | 56/56 requests healthy, acceptance 3.83 – 5.66 |
| TP2 | 4 | degraded; sustained-load mean 2.29 |
| TP1 | 8 | collapsed on every recycled-block request; sustained mean 1.89 |
| single node, no KV transfer | – | healthy at every parallelism tried |

**Version coverage.** Same signature on the 0.30 stack: fresh-block probes 5/7 healthy, recycled-block probes 7/8 collapsed at ~1.9.

**Architecture control.** A 64-KV-head MHA draft under the same PD geometry did not reproduce (12/12 healthy, 3.61 – 4.59), consistent with the root cause being on the grouped reformat path taken by the GQA draft group.

### Root cause

When decode TP < prefill TP, each decode rank pulls a KV shard in multiple slices and must reorganize heads across ranks after all pulls of that shard complete. The reorganization work is stashed per `(request_id, shard_idx)` and consumed in one batch:

- stash: [`_stash_pending_reformat`, L1091–L1098](https://github.com/vllm-project/vllm-ascend/blob/0bbf9248abdab3fa4256dc955bdb57abc06229e8/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1091-L1098)
- consume: [`_reformat_pending_kv_caches`, L1100–L1107](https://github.com/vllm-project/vllm-ascend/blob/0bbf9248abdab3fa4256dc955bdb57abc06229e8/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1100-L1107) — `pop`s the whole per-request dict once

A group is appended to `ready_attention_group_reformat_block_ids` only when its `is_group_transfer_end` flag is set (L1079–L1089). That flag is set on the **last pull of that group**:

- `is_group_transfer_end = remote_tp_offset == tp_num_need_pulls - 1` (L3630), and equivalents at L3704 and L3740 for the hybrid paths
- the per-group pull count is computed per group: `_get_attention_group_num_need_pulls_for_decode_tp` = `max(1, kv_heads // decode_tp) // max(1, kv_heads // prefill_tp)` (L3783–L3792)

Consequently a 1-KV-head group (MLA, indexer) has `pulls = 1` and is "transfer-end" on its **first** pull, while an 8-KV-head GQA group at prefill TP8 / decode TP1 has `pulls = 8` and is "transfer-end" only on its **eighth**. Groups of the same shard therefore become ready in **different pull tasks** — and those tasks run on different threads of the receiving `ThreadPoolExecutor` (L498), each calling `_stash_pending_reformat` once.

With the current implementation:

```python
with self.pending_reformat_lock:
    self.pending_reformat[request_id][shard_idx] = ready_attention_group_reformat_block_ids
```

the later stash **replaces** the whole bucket, so whichever groups arrived earlier are dropped and never reformatted. Which call lands last is a thread-scheduling race, which is why individual probes can occasionally look healthy while the defect is deterministic in aggregate.

Observed transfer-group split in the reproducing stack:

```
[('AscendSFAIndexerCacheSpec', 1, 1), ('AscendMLAAttentionSpec', 1, 1), ('FullAttentionSpec', 4, 8)]
```

Only the third group (the draft's GQA KV, 8 heads) needs cross-rank head reorganization — and it is exactly the one whose reformat gets dropped whenever a 1-pull group's stash lands after it.

### Why this stays invisible

- The target model's MLA / indexer groups have 1 KV head per transfer group, never need reorganization, and are always correct — **output text is unaffected**, so correctness monitoring sees nothing.
- Un-reformatted draft KV landing in zero-initialized blocks only mildly perturbs the draft (acceptance stays 4+); the collapse needs **block recycling**, which smoke tests with a fresh engine and a handful of requests never reach.
- Short-output scenarios do not trigger it either (the draft's sliding window naturally bounds short sequences), so routine correctness suites pass.

### Fix

Merge into the per-shard bucket instead of overwriting it — keep groups stashed earlier, skip duplicate arrivals of a group already stashed (dedup by `group_idx`):

```python
with self.pending_reformat_lock:
    stashed = self.pending_reformat[request_id].setdefault(shard_idx, [])
    stashed_group_ids = {entry[0] for entry in stashed}
    stashed.extend(
        entry
        for entry in ready_attention_group_reformat_block_ids
        if entry[0] not in stashed_group_ids
    )
```

When a shard becomes ready in a single batch (the only case where the old code was correct: decode TP ≥ prefill TP, or a single attention group) the bucket is empty and behavior is byte-identical to the current code. Lock scope and the consumer are unchanged.

### Before / after with the fix

Same probes, same two clusters, same request mix: **16/16 healthy** after the fix; recycled-block acceptance recovers from ~1.9 to **≥ 3.3**; the two-state (fresh vs recycled blocks) signature disappears.

### Related issues checked

#19463, #19502, #15586, #14703 describe PD-disaggregation symptoms in the same area but have different root causes (none touches the pending-reformat stash); in-flight PRs #18476, #18860, #17414, #19660, #17776, #13965 do not modify `_stash_pending_reformat` either.

A PR with the fix above plus a focused regression unit test follows: it will be linked here.
