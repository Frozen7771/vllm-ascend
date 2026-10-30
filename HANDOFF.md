# 接力包：vllm-ascend 上游 issue + PR 提交（只剩提交这一步）

## 已完成状态（不要重做）

- 修复分支已推上 fork：`Frozen7771/vllm-ascend` 分支 `fix/mooncake-stash-merge`，commit `84abc815db73ccaab0ce070cdf7e7ba6da779f68`，基于 upstream/main `0bbf9248abdab3fa4256dc955bdb57abc06229e8`。
- 改动：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` 的 `_stash_pending_reformat` 由整桶覆盖改为合并去重（setdefault 取桶 + 按 entry[0] 去重 extend）；`tests/ut/distributed/kv_transfer/kv_p2p/test_mooncake_connector.py` 尾部新增回归测试 `test_stash_pending_reformat_merges_batches_arriving_separately`。commit 已带 DCO sign-off。
- 已验证：pre-commit（ruff check / ruff format / codespell / typos / 仓内本地钩子）全部通过，formatter 零改动；对打好补丁的真函数体做过 4000 步随机差分对拍（单批与原版逐字节等价、多批合并去重符合预期）。
- 调研已定谳（一手核验）：缺陷在 main / releases/v0.26.0rc / releases/v0.30.0 / releases/v0.31.0rc 全部存在；引入自 PR #12359（commit `1db626510`）；无同根因 issue；在途 PR 无一触及该函数。

## 待办（本会话只做这三步，全部用本目录的定稿文本）

1. 用 add_repo 挂 `vllm-project/vllm-ascend`（access: `"push"`，用户会在界面批准）。不要挂 fork（与上游同名会检出冲突，而且不需要——分支已在 fork 上）。
2. 建 issue：
   ```bash
   gh issue create -R vllm-project/vllm-ascend --title "$(cat issue_title.txt)" --body-file issue_body.md
   ```
   记下返回的 issue 号 N。
3. 建 PR：先把 `pr_body.md` 里的 `{ISSUE_NUMBER}` 占位符（共 2 处）替换为 N，然后：
   ```bash
   gh pr create -R vllm-project/vllm-ascend --base main --head Frozen7771:fix/mooncake-stash-merge --title "$(cat pr_title.txt)" --body-file pr_body.md
   ```
   记下 PR 号 M。

## 不要做的事

- `issue_title.txt` / `issue_body.md` / `pr_title.txt` / `pr_body.md` / `fragment.yaml` 均为定稿：不得改写、增删内容，只允许替换 `{ISSUE_NUMBER}` / `{PR_NUMBER}` 占位符。
- 文本已按脱敏规则审定，不得加入任何内部主机名、集群代号、内部仓名、内部 MR 号等内部信息。
- release-note fragment（`fragment.yaml` → `release_notes/fragments/M.yaml`）不在本会话做：fork 的 push 通道在原会话。完成后把 M 告诉用户，由原会话推 fragment 跟进 commit。

## 回报

issue 号、PR 号、PR 链接。然后提醒用户：把 PR 号带回原会话，fragment 由那边推送；本 handoff 分支用完可删（原会话会处理）。
