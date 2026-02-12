"""Depth-wise batching simulation for recursive transformers.

Simulates the depth-wise batching inference strategy described in Bae et al.
(2024) Section 4.2, where tokens that exit early at different recursion depths
can be batched together to improve GPU utilization and throughput.

This is a simulation module that computes theoretical speedups rather than
implementing a real serving system.
"""

from dataclasses import dataclass

from src.inference.early_exit import EarlyExitStats


@dataclass
class BatchingSimulationResult:
    """Results from depth-wise batching simulation.

    Attributes:
        standard_total_ops: Total operations under standard (non-batched) inference.
        depthwise_total_ops: Total operations under depth-wise batching.
        theoretical_speedup: Ratio of standard to depth-wise operations.
        ops_per_depth: List of operation counts at each recursion depth.
        tokens_per_depth: List of token counts exiting at each depth.
    """

    standard_total_ops: float
    depthwise_total_ops: float
    theoretical_speedup: float
    ops_per_depth: list[float]
    tokens_per_depth: list[int]


def simulate_depth_wise_batching(
    exit_stats: EarlyExitStats,
    batch_size: int,
    num_loops: int,
) -> BatchingSimulationResult:
    """Simulate throughput improvement from depth-wise batching.

    In standard inference, every token must go through all num_loops recursive
    passes. With depth-wise batching, tokens that exit early free up compute
    for new tokens, improving overall throughput.

    Args:
        exit_stats: Early exit statistics from generation.
        batch_size: Batch size for the simulation.
        num_loops: Total number of recursive loops.

    Returns:
        BatchingSimulationResult with throughput comparison data.
    """
    tokens_per_depth = [0] * num_loops
    for depth in exit_stats.exit_depths:
        clamped_depth = min(depth, num_loops - 1)
        tokens_per_depth[clamped_depth] += 1

    total_tokens = sum(tokens_per_depth)
    standard_total_ops = float(total_tokens * num_loops)

    depthwise_total_ops = 0.0
    ops_per_depth = []
    for depth_idx in range(num_loops):
        tokens_at_this_depth = sum(tokens_per_depth[depth_idx:])
        depth_ops = float(tokens_at_this_depth)
        ops_per_depth.append(depth_ops)
        depthwise_total_ops += depth_ops

    if depthwise_total_ops == 0:
        theoretical_speedup = 1.0
    else:
        theoretical_speedup = standard_total_ops / depthwise_total_ops

    return BatchingSimulationResult(
        standard_total_ops=standard_total_ops,
        depthwise_total_ops=depthwise_total_ops,
        theoretical_speedup=theoretical_speedup,
        ops_per_depth=ops_per_depth,
        tokens_per_depth=tokens_per_depth,
    )


def format_batching_comparison(
    result: BatchingSimulationResult,
    num_loops: int,
) -> str:
    """Format batching simulation results as a markdown table.

    Args:
        result: Simulation results to format.
        num_loops: Number of recursive loops.

    Returns:
        Formatted markdown string with comparison table.
    """
    lines = [
        "| Metric | Standard | Depth-wise |",
        "|--------|----------|------------|",
        f"| Total Ops | {result.standard_total_ops:.0f} | {result.depthwise_total_ops:.0f} |",
        f"| Speedup | 1.00x | {result.theoretical_speedup:.2f}x |",
        "",
        "| Depth | Tokens Exiting | Ops at Depth |",
        "|-------|----------------|--------------|",
    ]

    for depth_idx in range(num_loops):
        lines.append(
            f"| {depth_idx} | {result.tokens_per_depth[depth_idx]} "
            f"| {result.ops_per_depth[depth_idx]:.0f} |"
        )

    return "\n".join(lines)
