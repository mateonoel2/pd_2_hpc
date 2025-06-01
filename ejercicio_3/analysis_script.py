import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_benchmark_data(csv_file="ejercicio_3/knn_benchmark_results.csv"):
    """Load and validate benchmark results"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} benchmark results")
        print(f"Process counts: {sorted(df['processes'].unique())}")
        print(
            f"Dataset info: {df['train_samples'].iloc[0]} training, {df['test_samples'].iloc[0]} test samples"
        )
        return df
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run the benchmark first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_theoretical_metrics(df):
    """Calculate theoretical speedup and efficiency metrics"""
    # Get baseline (serial) performance - use p=2 as reference since we don't have p=1
    baseline_row = df[df["processes"] == df["processes"].min()].iloc[0]

    # Theoretical metrics
    df = df.copy()
    df["theoretical_speedup"] = df["processes"]
    df["theoretical_efficiency"] = 100.0  # 100% efficiency

    # Calculate actual speedup relative to smallest process count
    baseline_time = baseline_row["max_comp_time"] * baseline_row["processes"]
    df["actual_speedup"] = baseline_time / df["max_comp_time"]
    df["actual_efficiency"] = (df["actual_speedup"] / df["processes"]) * 100

    return df


def create_speedup_analysis(df):
    """Generate comprehensive speedup analysis and plots"""

    # Calculate enhanced metrics
    df = calculate_theoretical_metrics(df)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("KNN Parallel Performance Analysis", fontsize=16, fontweight="bold")

    processes = df["processes"].values

    # 1. Speedup Graph
    ax1 = axes[0, 0]
    ax1.plot(
        processes,
        df["theoretical_speedup"],
        "k--",
        linewidth=2,
        label="Theoretical (Linear)",
        marker="o",
        markersize=6,
    )
    ax1.plot(
        processes, df["speedup"], "b-", linewidth=2, label="Actual Speedup", marker="s", markersize=6
    )

    ax1.set_xlabel("Number of Processes (p)")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Speedup vs Number of Processes")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(processes)

    # Add speedup values as annotations
    for i, (p, speedup) in enumerate(zip(processes, df["speedup"])):
        ax1.annotate(
            f"{speedup:.2f}x",
            (p, speedup),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # 2. Efficiency Graph
    ax2 = axes[0, 1]
    ax2.plot(
        processes,
        df["theoretical_efficiency"],
        "k--",
        linewidth=2,
        label="Theoretical (100%)",
        marker="o",
        markersize=6,
    )
    ax2.plot(
        processes,
        df["actual_efficiency"],
        "r-",
        linewidth=2,
        label="Actual Efficiency",
        marker="^",
        markersize=6,
    )

    ax2.set_xlabel("Number of Processes (p)")
    ax2.set_ylabel("Parallel Efficiency (%)")
    ax2.set_title("Parallel Efficiency vs Number of Processes")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(processes)
    ax2.set_ylim(0, 110)

    # 3. Execution Time Breakdown
    ax3 = axes[1, 0]
    width = 0.35
    x = np.arange(len(processes))

    ax3.bar(x - width / 2, df["max_comp_time"], width, label="Computation Time", alpha=0.8)
    ax3.bar(x + width / 2, df["comm_time"], width, label="Communication Time", alpha=0.8)

    ax3.set_xlabel("Number of Processes (p)")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Computation vs Communication Time")
    ax3.set_xticks(x)
    ax3.set_xticklabels(processes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Load Balance Analysis
    ax4 = axes[1, 1]
    load_imbalance = (df["max_comp_time"] - (df["avg_comp_time"])) / df["max_comp_time"] * 100

    ax4.bar(processes, load_imbalance, alpha=0.7, color="orange")
    ax4.set_xlabel("Number of Processes (p)")
    ax4.set_ylabel("Load Imbalance (%)")
    ax4.set_title("Load Imbalance vs Number of Processes")
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(processes)

    # Add percentage labels on bars
    for i, (p, imbalance) in enumerate(zip(processes, load_imbalance)):
        ax4.text(p, imbalance + 0.5, f"{imbalance:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("knn_speedup_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    return df


def create_flops_analysis(df):
    """Generate FLOP/s analysis and computational intensity plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("KNN FLOP/s Performance Analysis", fontsize=16, fontweight="bold")

    processes = df["processes"].values

    # 1. FLOP/s vs Number of Processes
    ax1 = axes[0, 0]
    flops_per_sec = df["flops_per_second"] / 1e6  # Convert to MFLOP/s

    ax1.plot(processes, flops_per_sec, "g-", linewidth=2, marker="o", markersize=8, label="Actual FLOP/s")

    # Theoretical linear scaling
    theoretical_flops = flops_per_sec.iloc[0] * processes / processes.iloc[0]
    ax1.plot(
        processes, theoretical_flops, "k--", linewidth=2, marker="s", markersize=6, label="Theoretical Linear"
    )

    ax1.set_xlabel("Number of Processes (p)")
    ax1.set_ylabel("MFLOP/s (Million FLOP/s)")
    ax1.set_title("FLOP/s Performance vs Number of Processes")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(processes)

    # Add FLOP/s values as annotations
    for i, (p, flops) in enumerate(zip(processes, flops_per_sec)):
        ax1.annotate(
            f"{flops:.1f}M", (p, flops), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )

    # 2. FLOP Efficiency
    ax2 = axes[0, 1]
    flop_efficiency = (flops_per_sec / theoretical_flops) * 100

    ax2.plot(
        processes, flop_efficiency, "purple", linewidth=2, marker="^", markersize=8, label="FLOP Efficiency"
    )
    ax2.axhline(y=100, color="k", linestyle="--", alpha=0.5, label="Theoretical (100%)")

    ax2.set_xlabel("Number of Processes (p)")
    ax2.set_ylabel("FLOP Efficiency (%)")
    ax2.set_title("FLOP Efficiency vs Number of Processes")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(processes)
    ax2.set_ylim(0, 110)

    # 3. Computational Intensity Analysis
    ax3 = axes[1, 0]

    # Calculate computational intensity (FLOPs per byte transferred)
    # Assuming double precision (8 bytes per number)
    n_features = df["features"].iloc[0]
    n_train = df["train_samples"].iloc[0]
    n_test = df["test_samples"].iloc[0]

    bytes_per_sample = n_features * 8  # 8 bytes per double
    total_data_bytes = (n_train + n_test) * bytes_per_sample
    computational_intensity = df["total_flops"] / total_data_bytes

    ax3.bar(processes, computational_intensity, alpha=0.7, color="cyan")
    ax3.set_xlabel("Number of Processes (p)")
    ax3.set_ylabel("Computational Intensity\n(FLOPs/Byte)")
    ax3.set_title("Computational Intensity vs Number of Processes")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(processes)

    # 4. Speedup vs FLOP/s Relationship
    ax4 = axes[1, 1]

    # Calculate actual speedup from the data
    baseline_time = df["max_comp_time"].iloc[0] * df["processes"].iloc[0]
    actual_speedup = baseline_time / df["max_comp_time"]

    # Scatter plot with trend line
    ax4.scatter(
        actual_speedup, flops_per_sec, s=100, alpha=0.7, c=processes, cmap="viridis", edgecolors="black"
    )

    # Add trend line
    z = np.polyfit(actual_speedup, flops_per_sec, 1)
    p = np.poly1d(z)
    ax4.plot(actual_speedup, p(actual_speedup), "r--", alpha=0.8, linewidth=2)

    ax4.set_xlabel("Speedup")
    ax4.set_ylabel("MFLOP/s")
    ax4.set_title("Speedup vs FLOP/s Relationship")
    ax4.grid(True, alpha=0.3)

    # Add colorbar for process count
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label("Number of Processes")

    # Add process labels to points
    for i, (speedup, flops, p) in enumerate(zip(actual_speedup, flops_per_sec, processes)):
        ax4.annotate(
            f"p={p}", (speedup, flops), textcoords="offset points", xytext=(5, 5), ha="left", fontsize=9
        )

    plt.tight_layout()
    plt.savefig("knn_flops_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    return computational_intensity


def generate_analysis_report(df, computational_intensity):
    """Generate comprehensive text analysis report"""

    print("\n" + "=" * 80)
    print("KNN PARALLEL PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)

    # Basic dataset info
    print("\nDATASET INFORMATION:")
    print(f"  Training samples (n): {df['train_samples'].iloc[0]:,}")
    print(f"  Test samples: {df['test_samples'].iloc[0]:,}")
    print(f"  Features (d): {df['features'].iloc[0]}")
    print(f"  K value: {df['k'].iloc[0]}")

    # FLOP analysis
    total_flops = df["total_flops"].iloc[0]
    print(f"\nFLOP ANALYSIS:")
    print(f"  Total FLOPs: {total_flops:,}")
    print(f"  FLOPs per distance calc: {3 * df['features'].iloc[0]}")
    print(f"  Total distance calculations: {df['train_samples'].iloc[0] * df['test_samples'].iloc[0]:,}")
    print(
        f"  FLOP formula: 3d × n × test_samples = {3 * df['features'].iloc[0]} × {df['train_samples'].iloc[0]} × {df['test_samples'].iloc[0]} = {total_flops:,}"
    )

    # Performance metrics
    print(f"\nPERFORMANCE METRICS BY PROCESS COUNT:")
    print(f"{'Processes':<10} {'Speedup':<10} {'Efficiency':<12} {'MFLOP/s':<12} {'Comp Time':<12}")
    print("-" * 66)

    for _, row in df.iterrows():
        baseline_time = df["max_comp_time"].iloc[0] * df["processes"].iloc[0]
        speedup = baseline_time / row["max_comp_time"]
        efficiency = (speedup / row["processes"]) * 100
        mflops = row["flops_per_second"] / 1e6

        print(
            f"{row['processes']:<10} {speedup:<10.2f} {efficiency:<12.1f}% {mflops:<12.1f} {row['max_comp_time']:<12.3f}s"
        )

    # Scaling analysis
    max_speedup = (df["max_comp_time"].iloc[0] * df["processes"].iloc[0]) / df["max_comp_time"].min()
    max_processes = df["processes"].max()

    print(f"\nSCALING ANALYSIS:")
    print(f"  Best speedup achieved: {max_speedup:.2f}x with {max_processes} processes")
    print(f"  Best efficiency: {(max_speedup / max_processes) * 100:.1f}%")
    print(f"  Peak FLOP/s: {df['flops_per_second'].max() / 1e6:.1f} MFLOP/s")

    # Communication overhead
    comm_overhead = df["comm_time"] / df["total_time"] * 100
    print(f"\nCOMMUNICATION OVERHEAD:")
    print(f"  Communication time percentage: {comm_overhead.mean():.1f}% (avg)")
    print(f"  Range: {comm_overhead.min():.1f}% - {comm_overhead.max():.1f}%")

    # Computational intensity
    print(f"\nCOMPUTATIONAL INTENSITY:")
    print(f"  Average: {computational_intensity.mean():.1f} FLOPs/Byte")
    print(
        f"  This indicates a {'compute-bound' if computational_intensity.mean() > 1 else 'memory-bound'} algorithm"
    )

    # Recommendations
    print(f"\nRECOMMENDations:")
    if max_speedup / max_processes > 0.8:
        print("  ✓ Good scaling efficiency - algorithm benefits well from parallelization")
    else:
        print("  ⚠ Limited scaling efficiency - consider optimizing communication or load balance")

    if comm_overhead.mean() > 20:
        print("  ⚠ High communication overhead - consider larger problem sizes per process")
    else:
        print("  ✓ Reasonable communication overhead")

    print("\n" + "=" * 80)


def main():
    """Main analysis function"""
    print("KNN Benchmark Results Analysis")
    print("=" * 40)

    # Load data
    df = load_benchmark_data()
    if df is None:
        return

    # Ensure data is sorted by process count
    df = df.sort_values("processes").reset_index(drop=True)

    # Generate speedup analysis
    print("\nGenerating speedup analysis...")
    df_enhanced = create_speedup_analysis(df)

    # Generate FLOP/s analysis
    print("Generating FLOP/s analysis...")
    computational_intensity = create_flops_analysis(df_enhanced)

    # Generate comprehensive report
    generate_analysis_report(df_enhanced, computational_intensity)

    # Save enhanced data
    df_enhanced.to_csv("ejercicio_3/knn_analysis_results.csv", index=False)
    print(f"\nEnhanced results saved to 'knn_analysis_results.csv'")
    print("Graphs saved as 'knn_speedup_analysis.png' and 'knn_flops_analysis.png'")


if __name__ == "__main__":
    main()
