from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# Load and split data (only on rank 0)
if rank == 0:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
else:
    X_train = y_train = X_test = y_test = None

# Broadcast all data to all processes
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# Divide test data among processes
test_size = len(X_test)
chunk_size = test_size // size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size if rank < size - 1 else test_size

# Each process works on a chunk of test data
local_test_chunk = X_test[start_idx:end_idx]

# Parameter
k = 3

# Measure execution time
start_time = MPI.Wtime()

# Each process makes predictions for its chunk of test data
local_predictions = [knn_predict(x, X_train, y_train, k) for x in local_test_chunk]

# Gather all predictions
all_predictions = comm.gather(local_predictions, root=0)
end_time = MPI.Wtime()

# Combine results and evaluate (only on rank 0)
if rank == 0:
    # Flatten the gathered predictions
    y_pred = []
    for chunk_preds in all_predictions:
        y_pred.extend(chunk_preds)
    
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Execution time (parallel): {end_time - start_time:.4f} sec")
    
    # Visualize 10 predictions
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
        ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
        ax.axis("off")
    plt.suptitle("Sample Predictions (Parallel KNN)")
    plt.tight_layout()
    plt.show()