from mpi4py import MPI
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import argparse

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def euclidean_distance(a, b):
    """Calculate euclidean distance and return FLOP count"""
    # FLOPs for euclidean distance calculation:
    # - Subtraction: d FLOPs (where d is the number of dimensions)
    # - Squaring: d FLOPs
    # - Sum: (d-1) FLOPs
    # - Square root: 1 FLOP
    # Total: 3d FLOPs per distance calculation
    distance = np.sqrt(np.sum((a - b) ** 2))
    flops = 3 * len(a)  # 3d FLOPs per distance calculation
    return distance, flops


def knn_predict(test_point, X_train, y_train, k):
    """Make KNN prediction and track FLOPs"""
    total_flops = 0
    distances = []

    # Calculate distance to each training point
    for x in X_train:
        dist, flops = euclidean_distance(test_point, x)
        distances.append(dist)
        total_flops += flops

    # Sorting operations (not counted as these are comparison-based)
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)

    return most_common[0][0], total_flops


def generate_scaled_data(data_multiplier=1, use_synthetic=False, n_samples=1000):
    """Generate or scale dataset based on parameters"""
    if use_synthetic:
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=64,  # Same as digits dataset
            n_informative=40,
            n_redundant=10,
            n_classes=10,
            random_state=42,
        )
    else:
        # Use digits dataset - fixed size, don't multiply
        digits = load_digits()
        X, y = digits.data, digits.target

        # For consistent dataset size across different process counts
        # We keep the original digits dataset without multiplication

    return X, y


def main():
    # Parse command line arguments (only on rank 0)
    if rank == 0:
        parser = argparse.ArgumentParser(description="Parallel KNN Benchmark")
        parser.add_argument(
            "--data-multiplier", type=int, default=1, help="Data multiplier (ignored for consistency)"
        )
        parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of digits")
        parser.add_argument(
            "--n-samples", type=int, default=5000, help="Number of synthetic samples to generate"
        )
        parser.add_argument("--k", type=int, default=3, help="K value for KNN")
        parser.add_argument(
            "--no-plot", action="store_true", help="Skip plotting (useful for large scale tests)"
        )

        args = parser.parse_args()
        config = {
            "data_multiplier": args.data_multiplier,  # Keep for compatibility
            "use_synthetic": args.synthetic,
            "n_samples": args.n_samples,
            "k": args.k,
            "no_plot": args.no_plot,
        }
    else:
        config = None

    # Broadcast configuration to all processes
    config = comm.bcast(config, root=0)

    # Start total execution timer
    total_start = MPI.Wtime()

    # Data loading and preparation (only on rank 0)
    data_prep_start = MPI.Wtime()
    if rank == 0:
        # Use consistent dataset size regardless of data_multiplier for proper scaling analysis
        X, y = generate_scaled_data(1, config["use_synthetic"], config["n_samples"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Dataset prepared:")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"  - Classes: {len(np.unique(y))}")
        print(f"  - Processes: {size}")
        print("-" * 50)
    else:
        X_train = y_train = X_test = y_test = None

    data_prep_end = MPI.Wtime()

    # Communication phase 1: Broadcast data
    comm_start = MPI.Wtime()
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)
    comm_end_phase1 = MPI.Wtime()

    # Divide test data among processes
    test_size = len(X_test)
    chunk_size = test_size // size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < size - 1 else test_size

    # Each process works on a chunk of test data
    local_test_chunk = X_test[start_idx:end_idx]
    local_test_size = len(local_test_chunk)

    if rank == 0:
        print("Test data distribution:")
        for p in range(size):
            p_start = p * chunk_size
            p_end = p_start + chunk_size if p < size - 1 else test_size
            print(f"  Process {p}: {p_end - p_start} test samples")

    # Computation phase: KNN predictions with FLOP counting
    comp_start = MPI.Wtime()
    local_predictions = []
    local_total_flops = 0

    for i, test_point in enumerate(local_test_chunk):
        pred, flops = knn_predict(test_point, X_train, y_train, config["k"])
        local_predictions.append(pred)
        local_total_flops += flops

        # Progress indicator for large datasets
        if rank == 0 and local_test_size > 100 and (i + 1) % max(1, local_test_size // 10) == 0:
            print(
                f"  Process {rank} progress: {i + 1}/{local_test_size} ({100 * (i + 1) / local_test_size:.1f}%)"
            )

    comp_end = MPI.Wtime()

    # Communication phase 2: Gather results
    comm_start_phase2 = MPI.Wtime()
    all_predictions = comm.gather(local_predictions, root=0)
    all_flops = comm.gather(local_total_flops, root=0)
    comm_end_phase2 = MPI.Wtime()

    total_end = MPI.Wtime()

    # Calculate timing metrics for each process
    local_times = {
        "data_prep": data_prep_end - data_prep_start if rank == 0 else 0,
        "comm_phase1": comm_end_phase1 - comm_start,
        "computation": comp_end - comp_start,
        "comm_phase2": comm_end_phase2 - comm_start_phase2,
        "total_execution": total_end - total_start,
    }

    # Gather timing information from all processes
    all_times = comm.gather(local_times, root=0)

    # Results processing and output (only on rank 0)
    if rank == 0:
        # Flatten the gathered predictions
        y_pred = []
        for chunk_preds in all_predictions:
            y_pred.extend(chunk_preds)

        y_pred = np.array(y_pred)
        accuracy = np.mean(y_pred == y_test)

        # Calculate total FLOPs
        total_flops = sum(all_flops)

        # Calculate timing statistics
        total_comm_time = np.mean([t["comm_phase1"] + t["comm_phase2"] for t in all_times])
        max_comp_time = np.max([t["computation"] for t in all_times])
        min_comp_time = np.min([t["computation"] for t in all_times])
        avg_comp_time = np.mean([t["computation"] for t in all_times])
        total_exec_time = total_end - total_start

        # FLOP/s calculations
        flops_per_second = total_flops / max_comp_time if max_comp_time > 0 else 0
        avg_flops_per_second = total_flops / avg_comp_time if avg_comp_time > 0 else 0

        # Speedup calculations (compared to serial execution time estimation)
        # Serial time estimate: computation time if run on 1 process
        estimated_serial_time = avg_comp_time * size
        speedup = estimated_serial_time / max_comp_time if max_comp_time > 0 else 0
        efficiency = speedup / size * 100 if size > 0 else 0

        # Print comprehensive results
        print("\n" + "=" * 60)
        print("PARALLEL KNN BENCHMARK RESULTS")
        print("=" * 60)
        print("Configuration:")
        print(f"  - Processes (p): {size}")
        print(f"  - Training samples (n): {X_train.shape[0]:,}")
        print(f"  - Test samples: {X_test.shape[0]:,}")
        print(f"  - Features (d): {X_train.shape[1]}")
        print(f"  - K value: {config['k']}")
        print(f"  - Synthetic data: {config['use_synthetic']}")

        print("\nAccuracy:")
        print(f"  - Model accuracy: {accuracy:.4f}")
        print(f"  - Correct predictions: {np.sum(y_pred == y_test)}/{len(y_test)}")

        print("\nFLOP Analysis:")
        print(f"  - Total FLOPs: {total_flops:,}")
        print(f"  - FLOPs per test sample: {total_flops / len(X_test):,.0f}")
        print(f"  - FLOPs per distance calculation: {3 * X_train.shape[1]}")
        print(f"  - Total distance calculations: {len(X_test) * len(X_train):,}")

        print("\nTiming Results:")
        print(f"  - Total execution time: {total_exec_time:.4f} sec")
        print(f"  - Data preparation time: {all_times[0]['data_prep']:.4f} sec")
        print(f"  - Communication time: {total_comm_time:.4f} sec")
        print(f"    * Phase 1 (broadcast): {np.mean([t['comm_phase1'] for t in all_times]):.4f} sec")
        print(f"    * Phase 2 (gather): {np.mean([t['comm_phase2'] for t in all_times]):.4f} sec")
        print("  - Computation time:")
        print(f"    * Maximum: {max_comp_time:.4f} sec")
        print(f"    * Minimum: {min_comp_time:.4f} sec")
        print(f"    * Average: {avg_comp_time:.4f} sec")

        print("\nPerformance Metrics:")
        print(f"  - Peak FLOP/s: {flops_per_second:,.0f}")
        print(f"  - Average FLOP/s: {avg_flops_per_second:,.0f}")
        print(f"  - Speedup: {speedup:.2f}x")
        print(f"  - Parallel efficiency: {efficiency:.2f}%")

        print("\nLoad Balance:")
        load_imbalance = (max_comp_time - min_comp_time) / max_comp_time * 100 if max_comp_time > 0 else 0
        print(f"  - Load imbalance: {load_imbalance:.2f}%")
        print(f"  - Computation efficiency: {min_comp_time / max_comp_time * 100:.2f}%")

        # Scalability info
        print("\nScalability Info:")
        print(f"  - Test samples per process: ~{test_size // size}")
        print(f"  - Memory per process: ~{X_train.nbytes / size / 1024**2:.1f} MB")

        # Save results to file for batch processing - UPDATED FORMAT
        results_line = f"{size},{X_train.shape[0]},{X_test.shape[0]},{X_train.shape[1]},{config['k']},{accuracy:.6f},{total_exec_time:.6f},{total_comm_time:.6f},{max_comp_time:.6f},{avg_comp_time:.6f},{total_flops},{flops_per_second:.0f},{speedup:.4f},{efficiency:.2f}\n"

        try:
            with open("ejercicio_3/knn_benchmark_results.csv", "a") as f:
                # Write header if file is new
                try:
                    with open("ejercicio_3/knn_benchmark_results.csv", "r") as check_f:
                        if len(check_f.read()) == 0:
                            f.write(
                                "processes,train_samples,test_samples,features,k,accuracy,total_time,comm_time,max_comp_time,avg_comp_time,total_flops,flops_per_second,speedup,efficiency\n"
                            )
                except FileNotFoundError:
                    f.write(
                        "processes,train_samples,test_samples,features,k,accuracy,total_time,comm_time,max_comp_time,avg_comp_time,total_flops,flops_per_second,speedup,efficiency\n"
                    )
                f.write(results_line)
        except Exception as e:
            print(f"Could not save results to CSV: {e}")

        # Visualization (only for smaller datasets or if not disabled)
        if not config["no_plot"] and len(X_test) <= 1000:
            try:
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                for i, ax in enumerate(axes.flat):
                    if i < len(X_test):
                        # For synthetic data, create a simple 8x8 visualization
                        if config["use_synthetic"]:
                            # Reshape first 64 features to 8x8 for visualization
                            img_data = X_test[i][:64].reshape(8, 8)
                        else:
                            img_data = X_test[i].reshape(8, 8)

                        ax.imshow(img_data, cmap="gray")
                        color = "green" if y_pred[i] == y_test[i] else "red"
                        ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}", color=color, fontweight="bold")
                        ax.axis("off")

                plt.suptitle(
                    f"Sample Predictions - Parallel KNN\n"
                    f"Accuracy: {accuracy:.3f}, Processes: {size}, "
                    f"Samples: {X_train.shape[0]:,}",
                    fontsize=12,
                )
                plt.tight_layout()
                plt.savefig(f"knn_predictions_p{size}_n{X_train.shape[0]}.png", dpi=150, bbox_inches="tight")
                plt.show()
            except Exception as e:
                print(f"Visualization skipped: {e}")
        elif config["no_plot"]:
            print("Visualization skipped (--no-plot flag)")
        else:
            print("Visualization skipped (dataset too large)")


if __name__ == "__main__":
    main()
