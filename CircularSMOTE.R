# Load required libraries
library(mlbench)
library(FNN)
library(caret)
library(class)
library(ggplot2)
library(dplyr)

# --- Circular SMOTE Functions ---

generate_random_point_in_sphere <- function(center, radius, dim) {
  # Generate a random direction
  direction <- rnorm(dim)
  norm <- sqrt(sum(direction^2))
  if (norm == 0) {
    direction <- rep(1, dim)
    norm <- sqrt(sum(direction^2))
  }
  unit_direction <- direction / norm
  r_rand <- runif(1)^(1/dim)
  return(center + r_rand * radius * unit_direction)
}

circular_smote <- function(X_minority, N = 100, k = 5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X_minority <- as.matrix(X_minority)
  n_minority <- nrow(X_minority)
  n_features <- ncol(X_minority)
  
  if (N < 100) {
    n_generate <- floor((N / 100) * n_minority)
    indices <- sample(1:n_minority, n_generate)
    X_minority <- X_minority[indices, , drop = FALSE]
    N <- 100
  }
  
  N_per_sample <- as.integer(N / 100)
  knn_result <- get.knn(X_minority, k = k + 1)
  
  synthetic_samples <- matrix(nrow = 0, ncol = n_features)
  for (i in 1:n_minority) {
    x <- X_minority[i, ]
    neighbor_indices <- knn_result$nn.index[i, -1]  # exclude the sample itself
    for (j in 1:N_per_sample) {
      if (length(neighbor_indices) == 0) break
      nn_index <- sample(neighbor_indices, 1)
      x_neighbor <- X_minority[nn_index, ]
      midpoint <- (x + x_neighbor) / 2
      radius <- sqrt(sum((x - x_neighbor)^2)) / 2
      synthetic_sample <- generate_random_point_in_sphere(midpoint, radius, n_features)
      synthetic_samples <- rbind(synthetic_samples, synthetic_sample)
    }
  }
  colnames(synthetic_samples) <- colnames(X_minority)
  return(synthetic_samples)
}

# --- Data Loading and Preprocessing ---

# Load Breast Cancer dataset from mlbench
data(BreastCancer)
bc <- BreastCancer

# Remove rows with missing values and the ID column
bc <- bc[complete.cases(bc), ]
bc <- bc[, -1]

# Convert features from factors to numeric (except the Class label)
for (col in names(bc)[-ncol(bc)]) {
  bc[[col]] <- as.numeric(as.character(bc[[col]]))
}

# Check class distribution (Classes: "benign" and "malignant")
print(table(bc$Class))

# Split into training (80%) and testing (20%) sets using caret's createDataPartition
set.seed(42)
trainIndex <- createDataPartition(bc$Class, p = 0.8, list = FALSE)
trainData <- bc[trainIndex, ]
testData <- bc[-trainIndex, ]

print("Training set distribution:")
print(table(trainData$Class))

# --- Identify Minority Class and Apply Circular SMOTE ---

# Assume the minority class is the one with fewer instances
counts <- table(trainData$Class)
minority_class <- names(which.min(counts))
majority_class <- names(which.max(counts))
cat("Minority class:", minority_class, "\nMajority class:", majority_class, "\n")

# Separate training data by class
train_min <- trainData[trainData$Class == minority_class, ]
train_maj <- trainData[trainData$Class == majority_class, ]

# Prepare the minority feature matrix (remove Class column)
X_train_min <- as.matrix(train_min[, -ncol(train_min)])

# Calculate oversampling percentage to balance classes
n_min <- nrow(train_min)
n_maj <- nrow(train_maj)
N_percentage <- ceiling(100 * (n_maj - n_min) / n_min)
cat("Oversampling percentage (N):", N_percentage, "\n")

# Apply Circular SMOTE to generate synthetic minority samples
X_syn <- circular_smote(X_train_min, N = N_percentage, k = 5, seed = 42)
syn_df <- as.data.frame(X_syn)
syn_df$Class <- minority_class

# Combine synthetic samples with original training data
train_resampled <- rbind(trainData, syn_df)
cat("Training set distribution after oversampling:\n")
print(table(train_resampled$Class))

# --- Model Training and Evaluation ---

# Prepare features and labels for classification
train_features <- train_resampled[, -ncol(train_resampled)]
train_labels <- train_resampled$Class
test_features <- testData[, -ncol(testData)]
test_labels <- testData$Class

# Scale features using caret's preProcess
preProcValues <- preProcess(train_features, method = c("center", "scale"))
train_features_scaled <- predict(preProcValues, train_features)
test_features_scaled <- predict(preProcValues, test_features)

# Train a k-NN classifier (k = 5) using the class package
pred <- knn(train = train_features_scaled, test = test_features_scaled,
            cl = train_labels, k = 5)

# Evaluate performance using caret's confusionMatrix (set the minority class as positive)
conf_mat <- confusionMatrix(pred, test_labels, positive = minority_class)
print(conf_mat)

# --- Comparison: Model Without Oversampling ---

pred_orig <- knn(train = predict(preProcValues, trainData[, -ncol(trainData)]),
                 test = test_features_scaled,
                 cl = trainData$Class, k = 5)
conf_mat_orig <- confusionMatrix(pred_orig, test_labels, positive = minority_class)
cat("Performance without oversampling:\n")
print(conf_mat_orig)

# --- Plot Class Distributions Before and After Oversampling ---

dist_before <- as.data.frame(table(trainData$Class))
names(dist_before) <- c("Class", "Count")
dist_before$Type <- "Before"

dist_after <- as.data.frame(table(train_resampled$Class))
names(dist_after) <- c("Class", "Count")
dist_after$Type <- "After"

dist_combined <- rbind(dist_before, dist_after)
ggplot(dist_combined, aes(x = Class, y = Count, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Training Set Class Distribution") +
  theme_minimal()
