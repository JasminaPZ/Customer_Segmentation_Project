from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

def scale_data(df):
    """Scale the features using RobustScaler."""
    scaler = RobustScaler()
    scaled_df = scaler.fit_transform(df)
    return scaled_df

def perform_kmeans(X, n_clusters):
    """Perform KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.inertia_

def perform_pca(X, n_components=2):
    """Reduce dimensionality using PCA."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def perform_dbscan(X, eps=0.5, min_samples=10):
    """Perform DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels
