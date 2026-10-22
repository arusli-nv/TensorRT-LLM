# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep a WideEP PyExecutor serving after peer processes or whole nodes die, one loss after another."""

from typing import Dict, List, Sequence, Tuple

# A set of ranks lost together, as a sorted tuple: one rank or a whole node.
FailureDomain = Tuple[int, ...]


def failure_domains(hosts: Sequence[str], failure_domain: str) -> List[FailureDomain]:
    """The rank sets to recover from: every rank but 0, then for ``node`` every node but rank 0's.

    Args:
        hosts: Per rank, the host it runs on.
        failure_domain: ``rank`` or ``node`` (FaultToleranceConfig.failure_domain).
    """
    domains = [(rank,) for rank in range(1, len(hosts))]
    if failure_domain == "node":
        nodes: Dict[str, List[int]] = {}
        for rank, host in enumerate(hosts):
            nodes.setdefault(host, []).append(rank)
        domains += [tuple(ranks) for host, ranks in nodes.items() if host != hosts[0]]
    return list(dict.fromkeys(domains))
