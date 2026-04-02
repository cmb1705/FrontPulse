"""Example usage of the Pipeline API for programmatic execution.

This script demonstrates how to use the Pipeline class for various workflows:
- Running the full pipeline
- Running individual phases
- Using coupling configuration
- Handling cached data

Run from repo root:
    python examples/pipeline_usage.py
"""

from pathlib import Path

from src.graph_build import CouplingConfig
from src.pipeline import Pipeline


def example_full_pipeline():
    """Example: Run complete pipeline from ingest to community detection."""
    print("\n" + "=" * 60)
    print("Example 1: Full Pipeline Execution")
    print("=" * 60)

    pipeline = Pipeline(
        config_path="config/datasources.yaml",
        schema_path="config/schema.yaml",
        slices_path="config/slices.yaml",
        outdir="data/out",
        ingest_dir="data/current_ingest",
        graphs_dir="data/current_graphs",
        graph_mode="cumulative",
        log_level="INFO"
    )

    # Run full pipeline
    results = pipeline.run(skip_ingest=False, run_communities=True)

    print("\nResults:")
    print(f"- Ingested {len(results.df)} records")
    print(f"- Created {len(results.slices)} slices")
    print(f"- Built {sum(len(v) for v in results.graphs.values())} graphs")

    if results.errors:
        print(f"- Errors: {results.errors}")


def example_individual_phases():
    """Example: Run pipeline phases individually for more control."""
    print("\n" + "=" * 60)
    print("Example 2: Individual Phase Execution")
    print("=" * 60)

    pipeline = Pipeline(
        config_path="config/datasources.yaml",
        schema_path="config/schema.yaml",
        slices_path="config/slices.yaml",
        outdir="data/out",
        ingest_dir="data/current_ingest",
        graphs_dir="data/current_graphs",
        graph_mode="cumulative",
        log_level="INFO"
    )

    # Phase 1: Ingest (use cache if available)
    print("\nPhase 1: Ingest")
    df = pipeline.ingest(skip_cache=True)
    print(f"Loaded {len(df)} records from cache")

    # Phase 2: Slice
    print("\nPhase 2: Slice")
    slices = pipeline.slice(df)
    print(f"Created {len(slices)} slices")
    for name, slice_df in slices.items():
        print(f"  - {name}: {len(slice_df)} records")

    # Phase 3: Build graphs
    print("\nPhase 3: Build Graphs")
    graphs = pipeline.build_graphs(df, mode="cumulative")
    print(f"Built {len(graphs.get('cumulative', []))} cumulative graphs")

    # Phase 4: Community detection
    print("\nPhase 4: Community Detection")
    communities = pipeline.detect_communities(mode="cumulative", resume=True)
    print(f"Detected {len(communities)} communities (if any)")


def example_with_coupling():
    """Example: Run pipeline with bibliographic coupling enabled."""
    print("\n" + "=" * 60)
    print("Example 3: Pipeline with Coupling")
    print("=" * 60)

    # Configure coupling
    coupling_config = CouplingConfig(
        enabled=True,
        alpha=1.0,
        beta=0.3,
        lambda_decay=0.15,
        min_shared_refs=5,
        min_coupling_score=0.05,
        cache_dir=Path("data/out/cache_coupling"),
        workers=4
    )

    pipeline = Pipeline(
        config_path="config/datasources.yaml",
        schema_path="config/schema.yaml",
        slices_path="config/slices.yaml",
        outdir="data/out",
        ingest_dir="data/current_ingest",
        graphs_dir="data/current_graphs",
        graph_mode="cumulative",
        coupling_config=coupling_config,
        log_level="INFO"
    )

    # Run with coupling
    results = pipeline.run(skip_ingest=True, run_communities=False)

    print("\nResults with coupling:")
    print(f"- Ingested {len(results.df)} records")
    print(f"- Built {sum(len(v) for v in results.graphs.values())} graphs with coupling")


def example_notebook_workflow():
    """Example: Workflow suitable for Jupyter notebooks with exploratory analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Notebook-Friendly Workflow")
    print("=" * 60)

    # Initialize pipeline
    pipeline = Pipeline(
        config_path="config/datasources.yaml",
        schema_path="config/schema.yaml",
        slices_path="config/slices.yaml",
        outdir="data/out",
        ingest_dir="data/current_ingest",
        graphs_dir="data/current_graphs",
        log_level="WARNING"  # Less verbose for notebooks
    )

    # Load cached data
    df = pipeline.ingest(skip_cache=True)

    # Explore the data
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nPublication years: {df['pub_year'].min()} - {df['pub_year'].max()}")
    print(f"Unique quarters: {df['pub_qtr'].nunique()}")

    # Build a single graph for exploration
    from src.graph_build import build_direct_citation_graph

    # Filter to recent quarter
    latest_quarter = df['pub_qtr'].max()
    df_recent = df[df['pub_qtr'] == latest_quarter]

    print(f"\nBuilding graph for {latest_quarter} ({len(df_recent)} records)...")
    G = build_direct_citation_graph(df_recent)

    print("Graph stats:")
    print(f"- Nodes: {G.number_of_nodes()}")
    print(f"- Edges: {G.number_of_edges()}")
    print(f"- Density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)):.4f}")

    # At this point, you could run network analysis, visualization, etc.


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Full pipeline", example_full_pipeline),
        "2": ("Individual phases", example_individual_phases),
        "3": ("With coupling", example_with_coupling),
        "4": ("Notebook workflow", example_notebook_workflow),
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        name, func = examples[sys.argv[1]]
        print(f"\nRunning: {name}")
        func()
    else:
        print("2YP Pipeline API Examples")
        print("=" * 60)
        print("\nUsage: python examples/pipeline_usage.py [example_number]")
        print("\nAvailable examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}: {name}")
        print("\nRun without arguments to see this menu.")
        print("Run with argument (e.g., '1') to execute that example.")
