import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

def plot_elbow_method(X, max_k=20):
    """Plot the elbow curve to find the optimal number of clusters."""
    inertia = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        _, inertia_k = perform_kmeans(X, k)
        inertia.append(inertia_k)
    
    knee = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker='o', color="teal", label="Inertia")
    plt.vlines(knee.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", colors='red', label=f"Elbow at k = {knee.knee}")
    plt.title("Elbow Method with kneed")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Optimal number of clusters (k): {knee.knee}")
    return knee.knee

def calculate_silhouette_score(X, labels):
    """Calculate the silhouette score for a given clustering result."""
    return silhouette_score(X, labels)
