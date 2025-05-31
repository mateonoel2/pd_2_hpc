from mpi4py import MPI
import numpy as np
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
k = 3

# Load and split data (only on rank 0)
if rank == 0:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    train_size = len(X_train)
else:
    X_train = y_train = X_test = y_test = None
    train_size = None

# Broadcast test data and labels
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# Divide training data
local_train_size = train_size // size
local_X = np.empty((local_train_size, 64), dtype='float64')  # 64 features for digits
local_y = np.empty(local_train_size, dtype='int')

# Timing
t_start = MPI.Wtime()
comm.Scatter(X_train, local_X, root=0)
comm.Scatter(y_train, local_y, root=0)
t_dist = MPI.Wtime()

# Local distance computation
local_predictions = []
for x in X_test:
    dists = np.array([euclidean_distance(x, local_x) for local_x in local_X])
    k_indices = dists.argsort()[:k]
    k_labels = local_y[k_indices]
    local_predictions.append((dists[k_indices], k_labels))
t_comp = MPI.Wtime()

# Gather all partial predictions
all_dists = comm.gather(local_predictions, root=0)
t_gather = MPI.Wtime()

# Final prediction on rank 0
if rank == 0:
    final_preds = []
    for i in range(len(X_test)):
        all_neighbors = []
        for proc_preds in all_dists:
            all_neighbors.extend(zip(proc_preds[i][0], proc_preds[i][1]))
        all_neighbors.sort(key=lambda x: x[0])
        top_k = [label for _, label in all_neighbors[:k]]
        final_pred = Counter(top_k).most_common(1)[0][0]
        final_preds.append(final_pred)

    final_preds = np.array(final_preds)
    accuracy = np.mean(final_preds == y_test)

    print(f"[Process Count: {size}]")
    print(f"Total Time       : {t_gather - t_start:.4f} sec")
    print(f"  - Distribution : {t_dist - t_start:.4f} sec")
    print(f"  - Computation  : {t_comp - t_dist:.4f} sec")
    print(f"  - Gathering    : {t_gather - t_comp:.4f} sec")
    print(f"Accuracy         : {accuracy:.4f}")

    # Visualize 10 predictions
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
        ax.set_title(f"Pred: {final_preds[i]}\nTrue: {y_test[i]}")
        ax.axis("off")
    plt.suptitle(f"Sample Predictions (Parallel KNN - {size} processes)")
    plt.tight_layout()
    # plt.show() 