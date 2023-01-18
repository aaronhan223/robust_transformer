import numpy as np
import pdb


def initialization(num_samples, num_cluster):
    cluster_init_size = [num_samples // num_cluster for _ in range(num_cluster)]
    density = np.ones(num_samples) / num_samples
    all_indices = np.arange(num_samples)
    cluster_indices = []
    for i in range(num_cluster):
        indices = np.random.choice(a=all_indices, size=cluster_init_size[i], replace=False, p=density)
        cluster_indices.append(np.sort(indices))
        if i == num_cluster - 1:
            break
        density[indices] = 0
        nonzero = density > 0
        density[nonzero] = 1 / np.sum(nonzero)
    return cluster_indices

indices = initialization(197, 5)
print(indices)