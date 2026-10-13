[None][feat] add a run-time rank mask and abort to the NVLinkOneSided all-to-all

## Description

When an EP peer dies, NVLinkOneSided dispatch and combine on the survivors spin on its completion flags until the device timeout traps and kills their CUDA context. CUDA graphs bake in kernel arguments, so a rank mask passed by value cannot change after capture. This PR adds an execution descriptor that the kernels read at run time, for a bounded cooperative abort and a live-rank mask, and fixed per-source slots that keep a dead rank's late writes out of live rows. It is PR 5 of the WideEP peer-loss recovery series and has no dependency.

`moe_a2a_dispatch` and `moe_a2a_combine` take an optional `execution_descriptor`, an int64 device tensor holding `MoeA2AFenceDescriptor`: the active-rank mask words and an abort record with `requested` and `status`. It is valid only on the fence path with `enable_rank_mask`. With a descriptor:

- Kernels read the mask at run time, so captured graphs see a mask that the host rewrites while no collective runs.
- Spin loops exit when `requested` is set, and a kernel that returns early records `status`. A real timeout still traps. An aborted combine withholds its readiness flag, so no peer pulls its outputs.
- Each source rank writes to a fixed slot laid out as `[source][payload][C rows]`, with C the static `max_num_tokens`. Main lays rows out by this step's tokens per rank, so addresses move between steps and a dead rank's late writes could land on live rows. The op rejects a runtime token count above C.
- A new `moeA2AGatherLocalKernel`, launched with PDL, copies the valid rows into the dense tensors the expert kernels expect. Rows of masked sources, rows past a source's count and every row of an aborted step get the invalid expert id and a zero payload. Dispatch therefore requires `expert_id_payload_index` and `invalid_token_expert_id`.

The PR also repairs the fake ops in `cpp_custom_ops.py`. On main, the fake `moe_a2a_dispatch` lacks `use_cft_counted_writes`, `expert_id_payload_index` and `invalid_token_expert_id`, and the fake `moe_a2a_combine` lacks `use_cft_counted_writes`, so later arguments bind to the wrong parameters under fake tensors. Both now match the C++ schema, including `execution_descriptor`.

FT-off impact: without a descriptor the kernels run as on main. A SASS comparison against main 55279c109f covers all 299 kernel instantiations. The 150 that run with rank masking off, the only ones a run without FT uses, are opcode-identical; only operand offsets differ, because the parameter structs grew. The other 149 are the rank-mask variants this PR changes. The gather kernel is the only new kernel.

Coordination: #19610 rewrites these kernels; we rebase after it merges and move the abort check into its `wait_round_flag`. #15895 proposes a competing abort ABI that passes a host epoch by value and does not work with graphs; we will align with its owner.

PR 6 binds the descriptor to the model. It also carries the two-process late-write test, which needs PR 6's workspace sizing. With the full series on DeepSeek-R1 at EP32, a kill timed to land during dispatch and combine made all 31 survivors abort their in-flight MoE collectives, and their requests completed 2.6 s after the kill with exact tokens. For comparison, survivors' requests complete 0.8 s after an idle kill and 1.4 s after a kill during decode.

## Test Coverage

`tests/unittest/_torch/moe/test_moe_a2a_abort.py` is new and added to the MoE component tests of `l0_b200`:

- `test_cancel_and_exclusion_on_replay` checks that one captured graph observes the abort record and the mask written before each replay: a pending abort releases both waits, and the same graph completes once the silent peer is masked.
- `test_registered_source_slots_preserve_excluded_state` checks that graphs of different batch sizes share the fixed source slots and leave an excluded source's storage untouched.
- `test_gather_bounds_received_count`, `test_source_slots_reject_runtime_over_capacity` and `test_descriptor_rejects_cft_counted_writes`.
- `test_descriptor_schema_and_fake_signatures` checks that the fake ops match the schema.
- `test_fence_timeout_still_traps` checks in a fresh interpreter that a wait on a silent peer without an abort request still traps at the timeout.

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
