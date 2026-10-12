[None][feat] let attention-DP routing skip absent ranks

## Description

Attention-DP routing assumes every TP rank answers every host exchange. After a recovered peer loss, gathers return `None` or a fill value at the departed rank's logical position, as PR 4 provides. This PR makes routing and the per-step ADP exchanges work on the ranks that are present. It is PR 8 of the WideEP peer-loss recovery series and depends on PR 4.

- `ADPRouter.gather_all_rank_states` drops absent ranks. Per-rank request and token counts become dicts keyed by logical rank, in `adp_router.py`, `request_utils.py` and the executor.
- Before any router runs, `ADPRouter.split_unservable_requests` removes the requests strictly pinned to a rank that is not serving, and `PyExecutor._fail_unservable_requests` answers them with the error "Attention DP rank N is not serving". Every rank routes the same requests, so every rank reaches the response collective. `can_process_attention_dp_request` lets such a request leave the waiting queue so that it can be failed. `DefaultADPRouter` load balances a relaxed pin to a rank that is not serving.
- Admission counts an absent rank as full, so it never exceeds what the serving ranks hold.
- The batch-size exchange in `_can_queue` passes a fill value, so an absent rank does not veto the forward, and the CUDA graph vote in `CUDAGraphRunner` ignores absent ranks.

Recovery requires `DefaultADPRouter` and rejects attention-DP balancing and a KV cache transceiver, so the balance exchange, the fetch-status gather and the KV pool rebalance stay as on main.

FT-off impact: with every rank present, all exchanges return the same values as before. One change is visible without FT, for every router: on main, a request strictly pinned to an `attention_dp_rank` at or above the TP size raises `IndexError` while the executor admits and routes new requests. It now gets an error response, and other requests keep routing. With `DefaultADPRouter`, a relaxed pin to such a rank, which also raised `IndexError`, is now load balanced.

## Test Coverage

New file `tests/unittest/_torch/executor/test_adp_absent_ranks.py`, split from the series' `test_peer_loss_recovery.py`. `l0_cpu` runs it through `unittest/_torch/executor`.

- `test_gather_all_rank_states_omits_failed_ranks` and `test_default_router_balances_over_survivors_by_logical_rank`
- `test_only_strict_pins_to_failed_rank_are_unservable` and `test_relaxed_pin_to_failed_rank_is_rebalanced`
- `test_cuda_graph_vote_ignores_failed_rank`
- `test_waiting_queue_releases_strict_pin_to_failed_rank` and `test_fail_unservable_requests_responds_from_rank0_only`

`tests/unittest/_torch/executor/test_adp_host_exchange.py` is updated for the fill value and `live_ranks`, and the disaggregation test fake `fake_dist.py` accepts `fill_value`. The existing `test_adp_router.py`, `test_request_utils.py` and `test_waiting_queue.py` cover the unchanged FT-off routing.

## PR Checklist

Please review the following before submitting your PR:
- PR description clearly explains what and why. If using CodeRabbit's summary, please make sure it makes sense.
- PR Follows [TRT-LLM CODING GUIDELINES](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CODING_GUIDELINES.md) to the best of your knowledge.
- Test cases are provided for new code paths (see [test instructions](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tests#1-how-does-the-ci-work))
- If PR introduces API changes, an appropriate PR label is added - either `api-compatible` or `api-breaking`. For `api-breaking`, include `BREAKING` in the PR title.
- Any new dependencies have been scanned for license and vulnerabilities
- [CODEOWNERS](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/CODEOWNERS) updated if ownership changes
- Documentation updated as needed
- Update [tava architecture diagram](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.github/tava_architecture_diagram.md) if there is a significant design change in PR.
- The reviewers assigned automatically/manually are appropriate for the PR.


- [ ] Please check this after reviewing the above items as appropriate for this PR.

## GitHub Bot Help

To see a list of available CI bot commands, please comment `/bot help`.
