"""
Memory profiling and monitoring utilities (PERF-3).

Provides functions and decorators for tracking memory usage and
issuing warnings when system memory is low.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


# Memory thresholds (in GB)
MIN_RECOMMENDED_MEMORY_GB = 8.0
LOW_MEMORY_WARNING_GB = 4.0
CRITICAL_MEMORY_GB = 2.0


def get_memory_info() -> dict[str, float]:
    """
    Get current system memory information.

    Returns:
        Dictionary with memory stats in GB:
        - total: Total physical memory
        - available: Available memory
        - used: Used memory
        - percent: Percentage of memory used
    """
    if not PSUTIL_AVAILABLE:
        return {}

    mem = psutil.virtual_memory()
    return {
        "total": mem.total / (1024 ** 3),
        "available": mem.available / (1024 ** 3),
        "used": mem.used / (1024 ** 3),
        "percent": mem.percent,
    }


def check_memory_availability(logger: logging.Logger | None = None) -> bool:
    """
    Check if sufficient memory is available and issue warnings if needed.

    Args:
        logger: Optional logger for warnings

    Returns:
        True if memory is sufficient, False if critically low
    """
    if not PSUTIL_AVAILABLE:
        if logger:
            logger.debug("psutil not available; skipping memory check")
        return True

    mem_info = get_memory_info()
    available_gb = mem_info.get("available", 0)

    if available_gb < CRITICAL_MEMORY_GB:
        if logger:
            logger.error(
                f"CRITICAL: Only {available_gb:.2f} GB of memory available. "
                f"This may cause the process to fail. Minimum recommended: {MIN_RECOMMENDED_MEMORY_GB} GB"
            )
        return False
    elif available_gb < LOW_MEMORY_WARNING_GB:
        if logger:
            logger.warning(
                f"LOW MEMORY: Only {available_gb:.2f} GB available. "
                f"Recommended: {MIN_RECOMMENDED_MEMORY_GB} GB. "
                f"Consider reducing dataset size or increasing system memory."
            )
        return True
    elif available_gb < MIN_RECOMMENDED_MEMORY_GB:
        if logger:
            logger.info(
                f"Memory available: {available_gb:.2f} GB "
                f"(Recommended: {MIN_RECOMMENDED_MEMORY_GB} GB)"
            )
        return True
    else:
        if logger:
            logger.debug(f"Memory check OK: {available_gb:.2f} GB available")
        return True


def log_memory_usage(logger: logging.Logger, context: str = ""):
    """
    Log current memory usage with optional context.

    Args:
        logger: Logger instance
        context: Optional context string (e.g., "After graph building")
    """
    if not PSUTIL_AVAILABLE:
        return

    mem_info = get_memory_info()
    ctx = f" [{context}]" if context else ""
    logger.info(
        f"Memory usage{ctx}: "
        f"{mem_info['used']:.2f} GB used, "
        f"{mem_info['available']:.2f} GB available "
        f"({mem_info['percent']:.1f}% used)"
    )


def memory_monitor(func: Callable | None = None, *, logger: logging.Logger | None = None) -> Callable:
    """
    Decorator to monitor memory usage before and after function execution.

    Usage:
        @memory_monitor
        def my_function():
            ...

        # Or with a logger:
        @memory_monitor(logger=my_logger)
        def my_function():
            ...

    Args:
        func: Function to decorate (when used without parentheses)
        logger: Optional logger for output

    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal logger

            if not PSUTIL_AVAILABLE:
                return f(*args, **kwargs)

            # Try to get logger from kwargs if not provided
            if logger is None:
                logger = kwargs.get('logger') or (args[0] if args and hasattr(args[0], 'info') else None)

            # Get memory before execution
            mem_before = get_memory_info()
            if logger:
                logger.debug(
                    f"Memory before {f.__name__}: "
                    f"{mem_before['used']:.2f} GB used, "
                    f"{mem_before['available']:.2f} GB available"
                )

            # Check if sufficient memory is available
            check_memory_availability(logger)

            # Execute function
            result = f(*args, **kwargs)

            # Get memory after execution
            mem_after = get_memory_info()
            mem_delta = mem_after['used'] - mem_before['used']
            if logger:
                logger.info(
                    f"Memory after {f.__name__}: "
                    f"{mem_after['used']:.2f} GB used "
                    f"({mem_delta:+.2f} GB delta), "
                    f"{mem_after['available']:.2f} GB available"
                )

            return result

        return wrapper

    # Handle both @memory_monitor and @memory_monitor(logger=...)
    if func is None:
        return decorator
    else:
        return decorator(func)


def profile_memory(enabled: bool = True) -> Callable:
    """
    Decorator to enable detailed memory profiling using memory_profiler.

    Note: This requires memory_profiler to be installed and can significantly
    slow down execution. Use only for debugging memory issues.

    Usage:
        @profile_memory(enabled=True)
        def my_function():
            ...

    Args:
        enabled: Whether to enable profiling

    Returns:
        Decorated function (or original if profiling disabled/unavailable)
    """
    def decorator(func: Callable) -> Callable:
        if enabled and MEMORY_PROFILER_AVAILABLE:
            return memory_profile(func)
        return func

    return decorator


def suggest_chunking_strategy(n_rows: int, available_gb: float) -> dict[str, Any]:
    """
    Suggest a chunking strategy based on dataset size and available memory.

    Args:
        n_rows: Number of rows in dataset
        available_gb: Available memory in GB

    Returns:
        Dictionary with chunking recommendations:
        - should_chunk: Whether chunking is recommended
        - chunk_size: Suggested chunk size
        - reason: Explanation
    """
    # Rough heuristic: assume ~1KB per row on average for metadata-rich datasets
    estimated_memory_gb = (n_rows * 1024) / (1024 ** 3)

    if n_rows > 100_000 and available_gb < MIN_RECOMMENDED_MEMORY_GB:
        chunk_size = min(50_000, int((available_gb / estimated_memory_gb) * n_rows))
        return {
            "should_chunk": True,
            "chunk_size": chunk_size,
            "reason": f"Dataset has {n_rows:,} rows with only {available_gb:.2f} GB available",
        }
    elif n_rows > 500_000:
        chunk_size = 100_000
        return {
            "should_chunk": True,
            "chunk_size": chunk_size,
            "reason": f"Very large dataset ({n_rows:,} rows); chunking recommended even with sufficient memory",
        }
    else:
        return {
            "should_chunk": False,
            "chunk_size": n_rows,
            "reason": f"Dataset size ({n_rows:,} rows) manageable with available memory ({available_gb:.2f} GB)",
        }


def suggest_worker_count_for_memory(
    available_gb: float,
    node_count: int,
    total_refs: int,
    max_workers: int,
    logger: logging.Logger | None = None
) -> tuple[int, str]:
    """
    Suggest optimal worker count based on available memory and workload size.

    Uses heuristics to balance parallelism with memory constraints. Reduces
    worker count when memory is limited to prevent OOM crashes and excessive
    swapping.

    Args:
        available_gb: Available system memory in GB
        node_count: Number of nodes in the graph
        total_refs: Total number of references across all nodes
        max_workers: Maximum workers requested by user
        logger: Optional logger for detailed diagnostics

    Returns:
        Tuple of (adjusted_worker_count, reason_string)

    Examples:
        >>> suggest_worker_count_for_memory(16.0, 10000, 500000, 12, None)
        (12, 'Sufficient memory (16.0 GB available)')

        >>> suggest_worker_count_for_memory(3.5, 50000, 2000000, 12, None)
        (1, 'Low memory (3.5 GB available) - single worker mode to prevent OOM')
    """
    if not PSUTIL_AVAILABLE:
        return max_workers, "psutil unavailable; using max workers"

    # Heuristic: Estimate memory needed for coupling calculation
    # Based on: nodes × refs per node × worker count × bytes per pair
    # Rough estimate: each pair needs ~50 bytes in shared_counts dict
    avg_refs_per_node = total_refs / max(node_count, 1)

    # Estimate coupling pairs: roughly (nodes with refs)^2 × overlap_factor
    # Overlap factor is typically 0.1-0.5 depending on field coherence
    estimated_pairs = (node_count * node_count * 0.2) / 1_000_000  # Conservative
    estimated_memory_gb = (estimated_pairs * 50) / (1024 ** 3)

    if logger:
        logger.debug(
            f"Memory estimation: {node_count:,} nodes, {total_refs:,} refs "
            f"(avg {avg_refs_per_node:.1f}/node), ~{estimated_pairs:.1f}M pairs, "
            f"~{estimated_memory_gb:.2f} GB estimated"
        )

    # Memory thresholds and worker scaling
    # Critical: < 4GB available → force 1 worker
    if available_gb < 4.0:
        reason = f"Low memory ({available_gb:.1f} GB available) - single worker mode to prevent OOM"
        return 1, reason

    # Warning: < 8GB available → reduce to 25% workers
    elif available_gb < 8.0:
        adjusted = max(1, max_workers // 4)
        reason = (
            f"Limited memory ({available_gb:.1f} GB available) - "
            f"reducing from {max_workers} to {adjusted} workers"
        )
        return adjusted, reason

    # Moderate: < 12GB available → reduce to 50% workers
    elif available_gb < 12.0:
        adjusted = max(1, max_workers // 2)
        reason = (
            f"Moderate memory ({available_gb:.1f} GB available) - "
            f"reducing from {max_workers} to {adjusted} workers"
        )
        return adjusted, reason

    # Check if estimated memory exceeds 60% of available
    elif estimated_memory_gb > (available_gb * 0.6):
        # Scale down proportionally
        safe_workers = max(1, int(max_workers * available_gb / (estimated_memory_gb * 2)))
        reason = (
            f"High estimated memory usage ({estimated_memory_gb:.1f} GB) - "
            f"reducing from {max_workers} to {safe_workers} workers"
        )
        return safe_workers, reason

    else:
        return max_workers, f"Sufficient memory ({available_gb:.1f} GB available)"


def suggest_graph_worker_count(
    available_gb: float,
    total_works: int,
    num_graphs: int,
    max_workers: int,
    coupling_workers: int = 0,
    logger: logging.Logger | None = None
) -> tuple[int, str]:
    """
    Suggest optimal graph worker count for parallel graph building based on memory.

    This function estimates memory requirements for building multiple graphs in parallel,
    accounting for both graph structure memory and nested coupling worker processes.

    Args:
        available_gb: Available system memory in GB
        total_works: Total number of works across all graphs
        num_graphs: Number of graphs to build
        max_workers: Maximum graph workers requested by user
        coupling_workers: Number of coupling workers per graph (if enabled)
        logger: Optional logger for detailed diagnostics

    Returns:
        Tuple of (adjusted_worker_count, reason_string)

    Examples:
        >>> # Sufficient memory for parallel graph building
        >>> suggest_graph_worker_count(32.0, 100000, 20, 4, 12, None)
        (4, 'Sufficient memory (32.0 GB available) for 4 parallel graphs')

        >>> # Memory constrained - reduce parallelism
        >>> suggest_graph_worker_count(6.0, 100000, 20, 4, 12, None)
        (2, 'Limited memory (6.0 GB available) - reducing from 4 to 2 graph workers')
    """
    if not PSUTIL_AVAILABLE:
        return max_workers, "psutil unavailable; using max graph workers"

    # Estimate memory per graph
    # Rough heuristic: ~100 bytes per node, ~50 bytes per edge (conservative)
    # Citation graphs typically have edge/node ratio of ~10-20
    avg_works_per_graph = total_works / max(num_graphs, 1)
    estimated_nodes_per_graph = avg_works_per_graph
    estimated_edges_per_graph = avg_works_per_graph * 15  # Conservative estimate

    # Memory breakdown per graph:
    # - Graph structure: nodes + edges
    # - DataFrame slice: ~1KB per work
    # - Coupling workers: additional overhead if enabled
    graph_structure_gb = ((estimated_nodes_per_graph * 100) + (estimated_edges_per_graph * 50)) / (1024 ** 3)
    dataframe_gb = (avg_works_per_graph * 1024) / (1024 ** 3)

    # If coupling enabled, account for worker process overhead
    # Each coupling worker adds ~500MB base overhead + shared memory for coupling data
    coupling_overhead_gb = 0.0
    if coupling_workers > 0:
        coupling_overhead_gb = (coupling_workers * 0.5) + (graph_structure_gb * 0.3)

    estimated_memory_per_graph = graph_structure_gb + dataframe_gb + coupling_overhead_gb

    if logger:
        logger.debug(
            f"Graph memory estimation: {num_graphs} graphs, "
            f"~{estimated_nodes_per_graph:.0f} nodes/graph, "
            f"~{estimated_memory_per_graph:.2f} GB/graph "
            f"(structure: {graph_structure_gb:.2f} GB, data: {dataframe_gb:.2f} GB, "
            f"coupling overhead: {coupling_overhead_gb:.2f} GB)"
        )

    # Memory thresholds for graph-level parallelization
    # More conservative than coupling workers due to nested parallelism

    # Critical: < 4GB available → force 1 graph worker
    if available_gb < 4.0:
        reason = f"Low memory ({available_gb:.1f} GB available) - single graph worker mode to prevent OOM"
        return 1, reason

    # Warning: < 8GB available → reduce to 1-2 workers
    elif available_gb < 8.0:
        adjusted = min(2, max(1, max_workers // 4))
        reason = (
            f"Limited memory ({available_gb:.1f} GB available) - "
            f"reducing from {max_workers} to {adjusted} graph workers"
        )
        return adjusted, reason

    # Moderate: < 16GB available → reduce to 50% workers
    elif available_gb < 16.0:
        adjusted = max(1, max_workers // 2)
        reason = (
            f"Moderate memory ({available_gb:.1f} GB available) - "
            f"reducing from {max_workers} to {adjusted} graph workers"
        )
        return adjusted, reason

    # Check if estimated total memory exceeds 50% of available
    # (more conservative than coupling since we have nested parallelism)
    total_estimated = estimated_memory_per_graph * max_workers
    if total_estimated > (available_gb * 0.5):
        # Scale down to use ~40% of available memory
        safe_workers = max(1, int((available_gb * 0.4) / estimated_memory_per_graph))
        reason = (
            f"High estimated memory usage ({total_estimated:.1f} GB) - "
            f"reducing from {max_workers} to {safe_workers} graph workers"
        )
        return safe_workers, reason

    else:
        return max_workers, f"Sufficient memory ({available_gb:.1f} GB available) for {max_workers} parallel graphs"
