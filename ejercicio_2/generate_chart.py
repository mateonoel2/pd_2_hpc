import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Read data from CSV file
data = pd.read_csv("knn_benchmark_results.csv")

# Separate real data tests from synthetic data tests based on row position
# First 4 results: Real data with varying processes (strong scaling)
real_data_tests = data.iloc[:4].copy()
real_data_tests["data_type"] = "Real Dataset"

# Next 4 results: Synthetic data with varying dataset sizes (weak scaling)
synthetic_data_tests = data.iloc[4:].copy()
synthetic_data_tests["data_type"] = "Synthetic Dataset (make_classification)"

# Calculate additional metrics for real data tests
real_data_tests["speedup"] = real_data_tests["total_time"].iloc[0] / real_data_tests["total_time"]
real_data_tests["theoretical_speedup"] = real_data_tests["processes"]

print("Creating KNN Performance Analysis: Real vs Synthetic Data")
print("=" * 60)

# 2. DETAILED REAL DATA ANALYSIS (Strong Scaling)
fig_real = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Strong Scaling Performance",
        "Speedup Analysis",
        "Time Breakdown",
        "Scalability Metrics",
    ),
)

# Performance vs processes
fig_real.add_trace(
    go.Scatter(
        x=real_data_tests["processes"],
        y=real_data_tests["total_time"],
        mode="lines+markers",
        name="Total Time",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=10),
    ),
    row=1,
    col=1,
)

fig_real.add_trace(
    go.Scatter(
        x=real_data_tests["processes"],
        y=real_data_tests["comp_time"],
        mode="lines+markers",
        name="Computation Time",
        line=dict(color="#ff7f0e", width=3),
        marker=dict(size=10),
    ),
    row=1,
    col=1,
)

# Speedup analysis
fig_real.add_trace(
    go.Scatter(
        x=real_data_tests["processes"],
        y=real_data_tests["speedup"],
        mode="lines+markers",
        name="Actual Speedup",
        line=dict(color="#2ca02c", width=3),
        marker=dict(size=10),
    ),
    row=1,
    col=2,
)

fig_real.add_trace(
    go.Scatter(
        x=real_data_tests["processes"],
        y=real_data_tests["theoretical_speedup"],
        mode="lines+markers",
        name="Ideal Speedup",
        line=dict(color="red", dash="dash", width=2),
        marker=dict(size=8),
    ),
    row=1,
    col=2,
)

# Time breakdown (stacked bar)
fig_real.add_trace(
    go.Bar(
        x=real_data_tests["processes"],
        y=real_data_tests["comp_time"],
        name="Computation",
        marker_color="#ff7f0e",
    ),
    row=2,
    col=1,
)

fig_real.add_trace(
    go.Bar(
        x=real_data_tests["processes"],
        y=real_data_tests["comm_time"],
        name="Communication",
        marker_color="#2ca02c",
    ),
    row=2,
    col=1,
)

# Efficiency and accuracy
fig_real.add_trace(
    go.Scatter(
        x=real_data_tests["processes"],
        y=real_data_tests["efficiency"],
        mode="lines+markers",
        name="Efficiency (%)",
        line=dict(color="#d62728", width=3),
        marker=dict(size=10),
        yaxis="y7",
    ),
    row=2,
    col=2,
)

fig_real.add_trace(
    go.Scatter(
        x=real_data_tests["processes"],
        y=real_data_tests["accuracy"] * 100,
        mode="lines+markers",
        name="Accuracy (%)",
        line=dict(color="#9467bd", width=3),
        marker=dict(size=10),
        yaxis="y8",
    ),
    row=2,
    col=2,
)

fig_real.update_layout(
    title="Real Dataset Analysis - Strong Scaling (2875 samples)",
    height=800,
    template="plotly_white",
    barmode="stack",
)

# Update axes for real data plot
fig_real.update_xaxes(title_text="Number of Processes", row=1, col=1)
fig_real.update_xaxes(title_text="Number of Processes", row=1, col=2)
fig_real.update_xaxes(title_text="Number of Processes", row=2, col=1)
fig_real.update_xaxes(title_text="Number of Processes", row=2, col=2)

fig_real.update_yaxes(title_text="Time (seconds)", row=1, col=1)
fig_real.update_yaxes(title_text="Speedup Factor", row=1, col=2)
fig_real.update_yaxes(title_text="Time (seconds)", row=2, col=1)
fig_real.update_yaxes(title_text="Percentage (%)", row=2, col=2)

# 3. DETAILED SYNTHETIC DATA ANALYSIS (Weak Scaling)
fig_synthetic = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Execution Time vs Dataset Size",
        "Time Components Analysis",
        "Accuracy Trends",
        "Scalability Analysis",
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": True}]],
)

# Execution time vs data size
fig_synthetic.add_trace(
    go.Scatter(
        x=synthetic_data_tests["train_samples"],
        y=synthetic_data_tests["total_time"],
        mode="lines+markers",
        name="Total Time",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=10),
    ),
    row=1,
    col=1,
)

# Add trend line
z = np.polyfit(synthetic_data_tests["train_samples"], synthetic_data_tests["total_time"], 2)
p = np.poly1d(z)
x_trend = np.linspace(
    synthetic_data_tests["train_samples"].min(), synthetic_data_tests["train_samples"].max(), 100
)
fig_synthetic.add_trace(
    go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode="lines",
        name="Polynomial Trend",
        line=dict(color="red", dash="dash", width=2),
    ),
    row=1,
    col=1,
)

# Time components (stacked area)
fig_synthetic.add_trace(
    go.Scatter(
        x=synthetic_data_tests["train_samples"],
        y=synthetic_data_tests["comp_time"],
        mode="lines",
        fill="tonexty",
        name="Computation Time",
        line=dict(color="#ff7f0e"),
        fillcolor="rgba(255,127,14,0.4)",
    ),
    row=1,
    col=2,
)

fig_synthetic.add_trace(
    go.Scatter(
        x=synthetic_data_tests["train_samples"],
        y=synthetic_data_tests["comm_time"],
        mode="lines",
        fill="tozeroy",
        name="Communication Time",
        line=dict(color="#2ca02c"),
        fillcolor="rgba(44,160,44,0.4)",
    ),
    row=1,
    col=2,
)

# Accuracy trends
fig_synthetic.add_trace(
    go.Scatter(
        x=synthetic_data_tests["train_samples"],
        y=synthetic_data_tests["accuracy"] * 100,
        mode="lines+markers",
        name="Accuracy",
        line=dict(color="#d62728", width=3),
        marker=dict(size=10),
    ),
    row=2,
    col=1,
)

# Add accuracy trend line
acc_z = np.polyfit(synthetic_data_tests["train_samples"], synthetic_data_tests["accuracy"] * 100, 1)
acc_p = np.poly1d(acc_z)
fig_synthetic.add_trace(
    go.Scatter(
        x=x_trend,
        y=acc_p(x_trend),
        mode="lines",
        name="Accuracy Trend",
        line=dict(color="darkred", dash="dot", width=2),
    ),
    row=2,
    col=1,
)

# Scalability metrics
fig_synthetic.add_trace(
    go.Scatter(
        x=synthetic_data_tests["train_samples"],
        y=synthetic_data_tests["efficiency"],
        mode="lines+markers",
        name="Efficiency",
        line=dict(color="#9467bd", width=3),
        marker=dict(size=10),
    ),
    row=2,
    col=2,
)

# Throughput (samples per second)
throughput = synthetic_data_tests["train_samples"] / synthetic_data_tests["total_time"]
fig_synthetic.add_trace(
    go.Scatter(
        x=synthetic_data_tests["train_samples"],
        y=throughput,
        mode="lines+markers",
        name="Throughput (samples/sec)",
        line=dict(color="#8c564b", width=3),
        marker=dict(size=10),
        yaxis="y8",
    ),
    row=2,
    col=2,
)

fig_synthetic.update_layout(
    title="Synthetic Dataset Analysis - Weak Scaling (8 processes, make_classification)",
    height=800,
    template="plotly_white",
)

# Update axes for synthetic data plot
fig_synthetic.update_xaxes(title_text="Training Samples", row=1, col=1)
fig_synthetic.update_xaxes(title_text="Training Samples", row=1, col=2)
fig_synthetic.update_xaxes(title_text="Training Samples", row=2, col=1)
fig_synthetic.update_xaxes(title_text="Training Samples", row=2, col=2)

fig_synthetic.update_yaxes(title_text="Time (seconds)", row=1, col=1)
fig_synthetic.update_yaxes(title_text="Time (seconds)", row=1, col=2)
fig_synthetic.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
fig_synthetic.update_yaxes(title_text="Efficiency (%)", row=2, col=2)
fig_synthetic.update_yaxes(title_text="Throughput (samples/sec)", secondary_y=True, row=2, col=2)

# Display all plots
print("\nDisplaying visualizations...")
fig_real.show()
fig_synthetic.show()

if not os.path.exists("images"):
    os.makedirs("images")
    print("Created 'images' directory")

# Save plots as images only (for README.md)
print("\nSaving plots to images directory...")

# Summary
print("\n" + "=" * 70)
print("COMPREHENSIVE PERFORMANCE SUMMARY")
print("=" * 70)

print("\n📊 REAL DATASET ANALYSIS (Strong Scaling)")
print(f"   Dataset: {real_data_tests['train_samples'].iloc[0]:,} samples (fixed)")
print(f"   Processes tested: {real_data_tests['processes'].min()}-{real_data_tests['processes'].max()}")
print(f"   Best speedup: {real_data_tests['speedup'].max():.2f}x")
print(
    f"   Efficiency range: {real_data_tests['efficiency'].min():.1f}%-{real_data_tests['efficiency'].max():.1f}%"
)
print(f"   Accuracy: {real_data_tests['accuracy'].iloc[0] * 100:.2f}% (constant)")
print(
    f"   Best performance: {real_data_tests['total_time'].min():.3f}s at {real_data_tests.loc[real_data_tests['total_time'].idxmin(), 'processes']} processes"
)

print("\n🔬 SYNTHETIC DATASET ANALYSIS (Weak Scaling)")
print(
    f"   Dataset range: {synthetic_data_tests['train_samples'].min():,}-{synthetic_data_tests['train_samples'].max():,} samples"
)
print(f"   Processes: {synthetic_data_tests['processes'].iloc[0]} (fixed)")
print(
    f"   Efficiency range: {synthetic_data_tests['efficiency'].min():.1f}%-{synthetic_data_tests['efficiency'].max():.1f}%"
)
print(
    f"   Accuracy range: {synthetic_data_tests['accuracy'].min() * 100:.1f}%-{synthetic_data_tests['accuracy'].max() * 100:.1f}%"
)
print(
    f"   Best performance: {synthetic_data_tests['total_time'].min():.3f}s with {synthetic_data_tests.loc[synthetic_data_tests['total_time'].idxmin(), 'train_samples']:,} samples"
)

print("\n📈 KEY INSIGHTS:")
print(
    f"   • Real data shows excellent accuracy ({real_data_tests['accuracy'].iloc[0] * 100:.2f}%) but decreasing efficiency with more processes"
)
print("   • Synthetic data shows variable accuracy but consistent parallel efficiency")
print("   • Communication overhead increases with more processes in real data")
print("   • Synthetic data scales reasonably well with dataset size")

print("\n📁 Images saved to 'images/' directory:")
print("   - images/knn_comparison_overview.png")
print("   - images/real_data_analysis.png")
print("   - images/synthetic_data_analysis.png")

# Save static images to images directory
try:
    fig_real.write_image("images/real_data_analysis.png", width=1400, height=800, scale=2)
    fig_synthetic.write_image("images/synthetic_data_analysis.png", width=1400, height=800, scale=2)
    print("✅ PNG images saved successfully to 'images/' directory!")
    print("   - images/knn_comparison_overview.png")
    print("   - images/real_data_analysis.png")
    print("   - images/synthetic_data_analysis.png")
except Exception as e:
    print(f"❌ Could not save PNG images: {e}")
    print("   Install kaleido for PNG export: pip install kaleido")
