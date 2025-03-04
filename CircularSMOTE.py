import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

def generate_random_point_in_sphere(center, radius, dim):
    """
    Generate a random point uniformly within a d-dimensional sphere.
    """
    direction = np.random.normal(0, 1, dim)
    norm = np.linalg.norm(direction)
    if norm == 0:
        direction = np.ones(dim)
        norm = np.linalg.norm(direction)
    unit_direction = direction / norm
    # Uniform sampling in the sphere volume
    r_rand = radius * (np.random.rand() ** (1.0 / dim))
    return center + r_rand * unit_direction

def circular_smote(X_minority, N=100, k=5, random_state=None):
    """
    Apply Circular SMOTE oversampling.
    
    Parameters:
      X_minority: NumPy array of minority class samples.
      N: Oversampling percentage (e.g., 100 means 1 synthetic sample per original sample).
      k: Number of nearest neighbors.
      random_state: Seed for reproducibility.
      
    Returns:
      Synthetic samples as a NumPy array.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_minority, n_features = X_minority.shape

    # Handle the case when N < 100 (partial oversampling)
    if N < 100:
        n_generate = int((N / 100) * n_minority)
        if n_generate == 0:
            return np.empty((0, n_features))
        # Randomly select a subset of minority samples
        indices = np.random.choice(n_minority, n_generate, replace=False)
        X_minority = X_minority[indices]
        # Update the shape after sub-sampling
        n_minority, n_features = X_minority.shape
        # Reset N to 100 so that we generate one synthetic sample per sample
        N = 100

    # Calculate number of synthetic samples per original minority sample
    N_per_sample = int(N / 100)

    # Fit k-nearest neighbors (using k+1 since the point itself is included)
    neigh = NearestNeighbors(n_neighbors=k + 1)
    neigh.fit(X_minority)

    synthetic_samples = []

    for i in range(n_minority):
        x = X_minority[i]
        # Get neighbors and exclude the sample itself
        nn_indices = neigh.kneighbors([x], return_distance=False)[0]
        nn_indices = nn_indices[nn_indices != i]
        
        for _ in range(N_per_sample):
            if len(nn_indices) == 0:
                break
            nn_index = np.random.choice(nn_indices)
            x_neighbor = X_minority[nn_index]
            
            # Define the synthetic region using the midpoint and half the distance
            midpoint = (x + x_neighbor) / 2.0
            radius = np.linalg.norm(x - x_neighbor) / 2.0
            
            synthetic_sample = generate_random_point_in_sphere(midpoint, radius, n_features)
            synthetic_samples.append(synthetic_sample)
    
    return np.array(synthetic_samples)

def run_pipeline():
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    print("Original dataset distribution:", Counter(y))
    
    # Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training set distribution:", Counter(y_train))
    
    # Identify the minority class (assumed to be the class with fewer samples)
    counts = Counter(y_train)
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    print("Minority class:", minority_class, "Majority class:", majority_class)
    
    # Separate minority and majority samples from training set
    X_train_min = X_train[y_train == minority_class]
    X_train_maj = X_train[y_train == majority_class]
    
    # Compute oversampling percentage to balance classes:
    n_min = X_train_min.shape[0]
    n_maj = X_train_maj.shape[0]
    N_percentage = int(np.ceil(100 * (n_maj - n_min) / n_min))
    print("Oversampling percentage (N):", N_percentage)
    
    # Apply Circular SMOTE on minority samples
    X_syn = circular_smote(X_train_min, N=N_percentage, k=5, random_state=42)
    
    # Combine synthetic samples with the original training data
    X_train_resampled = np.vstack((X_train, X_syn))
    y_train_resampled = np.hstack((y_train, np.array([minority_class] * len(X_syn))))
    print("Resampled training set distribution:", Counter(y_train_resampled))
    
    # Optional: Feature selection using ANOVA F-value (using all features here)
    selector = SelectKBest(score_func=f_classif, k='all')
    X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
    X_test_selected = selector.transform(X_test)
    
    # Train a KNN classifier on the resampled data
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_selected, y_train_resampled)
    y_pred = clf.predict(X_test_selected)
    
    # Evaluate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=minority_class)
    recall = recall_score(y_test, y_pred, pos_label=minority_class)
    f1 = f1_score(y_test, y_pred, pos_label=minority_class)
    
    print("\nEvaluation Metrics (With Circular SMOTE):")
    print("Accuracy:", accuracy)
    print("Precision (minority):", precision)
    print("Recall (minority):", recall)
    print("F1 Score (minority):", f1)
    
    # For comparison: Train classifier without oversampling
    clf_orig = KNeighborsClassifier(n_neighbors=5)
    selector_orig = SelectKBest(score_func=f_classif, k='all')
    X_train_orig_selected = selector_orig.fit_transform(X_train, y_train)
    X_test_orig_selected = selector_orig.transform(X_test)
    clf_orig.fit(X_train_orig_selected, y_train)
    y_pred_orig = clf_orig.predict(X_test_orig_selected)
    
    accuracy_orig = accuracy_score(y_test, y_pred_orig)
    precision_orig = precision_score(y_test, y_pred_orig, pos_label=minority_class)
    recall_orig = recall_score(y_test, y_pred_orig, pos_label=minority_class)
    f1_orig = f1_score(y_test, y_pred_orig, pos_label=minority_class)
    
    print("\nEvaluation Metrics (Without Oversampling):")
    print("Accuracy:", accuracy_orig)
    print("Precision (minority):", precision_orig)
    print("Recall (minority):", recall_orig)
    print("F1 Score (minority):", f1_orig)
    
    # Plot training class distributions before and after oversampling
    labels = [str(majority_class), str(minority_class)]
    counts_before = [np.sum(y_train == majority_class), np.sum(y_train == minority_class)]
    counts_after = [np.sum(y_train_resampled == majority_class), np.sum(y_train_resampled == minority_class)]
    
    x_axis = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x_axis - width/2, counts_before, width, label='Before Oversampling')
    ax.bar(x_axis + width/2, counts_after, width, label='After Oversampling')
    ax.set_ylabel('Counts')
    ax.set_title('Training Set Class Distribution')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    run_pipeline()
