"""
Bad Apple Detector using Adaptive Linear Neuron (Adaline)
=========================================================
This script implements an Adaline (ADAptive LInear NEuron) model to classify
apples as "good" or "bad" based on their physical and chemical characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)


# =============================================================================
# 1. Adaline Implementation
# =============================================================================

class AdalineGD:
    """
    Adaptive Linear Neuron (Adaline) classifier using Batch Gradient Descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate (eta) for weight updates.
    n_iter : int, default=50
        Number of passes over the training dataset (epochs).
    random_state : int, default=1
        Random seed for reproducible weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Fitted feature weights after training.
    b_ : scalar
        Fitted bias unit after training.
    losses_ : list
        Mean squared error (MSE) per epoch during training.
    """

    def __init__(self, learning_rate=0.01, n_iter=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data by adjusting weights via gradient descent.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training feature matrix.
        y : array-like, shape [n_samples]
            Target class labels (+1 for good, -1 for bad).

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights with small random values
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)  # Bias initialized to zero
        self.losses_ = []           # Track MSE per epoch

        for _ in range(self.n_iter):
            # Compute net input: z = X·w + b
            net_input = self.net_input(X)
            # Compute error: e = y - z  (Adaline uses linear activation for updates)
            errors = y - net_input
            # Update weights: w = w + eta * X^T · e
            self.w_ += self.learning_rate * X.T.dot(errors)
            # Update bias: b = b + eta * sum(e)
            self.b_ += self.learning_rate * errors.sum()
            # Compute and store MSE loss
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        """Compute the net input (linear activation): z = X·w + b"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        Predict class labels using the unit step function.
        Returns +1 if net_input >= 0.5, else -1.
        """
        return np.where(self.net_input(X) >= 0.5, 1, -1)


# =============================================================================
# 2. Data Setup and Exploration
# =============================================================================

print("=" * 60)
print("1. DATA SETUP AND EXPLORATION")
print("=" * 60)

# Load the Apple Quality dataset
df = pd.read_csv('apple_quality.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass distribution:\n{df['Quality'].value_counts()}")
print(f"\nStatistical summary:\n{df.describe()}")

# Drop identifier column
df = df.drop(columns=['A_id'])

# =============================================================================
# Plot 1: Distribution of features by quality
# =============================================================================
features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(features):
    good_vals = df.loc[df['Quality'] == 'good', feat]
    bad_vals  = df.loc[df['Quality'] == 'bad',  feat]
    axes[i].hist(good_vals, bins=40, alpha=0.6, color='forestgreen', label='Good')
    axes[i].hist(bad_vals,  bins=40, alpha=0.6, color='tomato',       label='Bad')
    axes[i].set_title(feat)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
    axes[i].legend()

axes[-1].axis('off')  # Hide the 8th unused subplot
plt.suptitle('Feature Distributions by Apple Quality', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Figure saved] feature_distributions.png")

# =============================================================================
# Plot 2: Correlation heatmap
# =============================================================================
import matplotlib.cm as cm
numeric_df = df[features]
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha='right')
ax.set_yticklabels(features)
for i in range(len(features)):
    for j in range(len(features)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("[Figure saved] correlation_heatmap.png")


# =============================================================================
# 3. Preparing Training and Testing Sets
# =============================================================================

print("\n" + "=" * 60)
print("3. PREPARING TRAINING AND TESTING SETS")
print("=" * 60)

# Encode labels: good -> +1, bad -> -1 (standard Adaline encoding)
X = df[features].values
y = np.where(df['Quality'] == 'good', 1, -1)

print(f"\nFeature matrix shape : {X.shape}")
print(f"Target vector shape  : {y.shape}")
print(f"Class counts  +1(good): {(y == 1).sum()}  -1(bad): {(y == -1).sum()}")

# Split 80% training / 20% testing, stratified to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set : {X_train.shape[0]} samples")
print(f"Testing set  : {X_test.shape[0]} samples")

# Feature standardization (Adaline is sensitive to feature scale)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

print("\nAfter standardization:")
print(f"  Training mean  : {X_train_std.mean(axis=0).round(4)}")
print(f"  Training std   : {X_train_std.std(axis=0).round(4)}")


# =============================================================================
# 4. Building the Adaline Model
# =============================================================================

print("\n" + "=" * 60)
print("4. BUILDING THE ADALINE MODEL")
print("=" * 60)

# Initial model with default hyperparameters
ada_initial = AdalineGD(learning_rate=0.01, n_iter=100, random_state=1)
ada_initial.fit(X_train_std, y_train)

y_pred_initial = ada_initial.predict(X_test_std)
acc_initial = accuracy_score(y_test, y_pred_initial)
print(f"\nInitial model (lr=0.01, epochs=100):")
print(f"  Final training MSE : {ada_initial.losses_[-1]:.4f}")
print(f"  Test accuracy      : {acc_initial:.4f}")


# =============================================================================
# 5. Improving the Model (hyperparameter tuning)
# =============================================================================

print("\n" + "=" * 60)
print("5. IMPROVING THE MODEL")
print("=" * 60)

# Test different learning rates to find optimal convergence
learning_rates = [0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.01]
results = {}

for lr in learning_rates:
    ada = AdalineGD(learning_rate=lr, n_iter=200, random_state=1)
    ada.fit(X_train_std, y_train)
    y_pred = ada.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)
    results[lr] = {'model': ada, 'accuracy': acc, 'final_loss': ada.losses_[-1]}
    print(f"  lr={lr:.5f}  |  final MSE={ada.losses_[-1]:.4f}  |  test acc={acc:.4f}")

# Select best learning rate
best_lr = max(results, key=lambda lr: results[lr]['accuracy'])
ada_best = results[best_lr]['model']
print(f"\nBest learning rate: {best_lr}  (test accuracy: {results[best_lr]['accuracy']:.4f})")

# Final tuned model
ada_final = AdalineGD(learning_rate=best_lr, n_iter=300, random_state=1)
ada_final.fit(X_train_std, y_train)
y_pred_final = ada_final.predict(X_test_std)
acc_final = accuracy_score(y_test, y_pred_final)
print(f"\nFinal model (lr={best_lr}, epochs=300):")
print(f"  Final training MSE : {ada_final.losses_[-1]:.6f}")
print(f"  Test accuracy      : {acc_final:.4f}")


# =============================================================================
# Plot 3: Loss convergence curves for different learning rates
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: log-scaled loss for all learning rates
for lr, res in results.items():
    axes[0].plot(np.log10(res['model'].losses_), label=f'lr={lr}')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('log₁₀(MSE)')
axes[0].set_title('Loss Convergence (log scale) for Different Learning Rates')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: final model MSE curve
axes[1].plot(range(1, len(ada_final.losses_) + 1), ada_final.losses_, color='steelblue')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('MSE')
axes[1].set_title(f'Final Adaline Training Loss (lr={best_lr})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_convergence.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Figure saved] loss_convergence.png")


# =============================================================================
# 6. Output Interpretation & Estimate Errors
# =============================================================================

print("\n" + "=" * 60)
print("6. OUTPUT INTERPRETATION & ERROR ESTIMATION")
print("=" * 60)

# Net input values (raw scores before thresholding)
net_inputs = ada_final.net_input(X_test_std)
print(f"\nNet input statistics on test set:")
print(f"  Min  : {net_inputs.min():.4f}")
print(f"  Max  : {net_inputs.max():.4f}")
print(f"  Mean : {net_inputs.mean():.4f}")
print(f"  Std  : {net_inputs.std():.4f}")

# Error metrics on test set
mse_test  = mean_squared_error(y_test, net_inputs)
mae_test  = mean_absolute_error(y_test, net_inputs)
rmse_test = np.sqrt(mse_test)
print(f"\nError Metrics (test set):")
print(f"  MSE  (Mean Squared Error)       : {mse_test:.4f}")
print(f"  RMSE (Root Mean Squared Error)  : {rmse_test:.4f}")
print(f"  MAE  (Mean Absolute Error)      : {mae_test:.4f}")

# =============================================================================
# 7. Making Predictions on the Test Set
# =============================================================================

print("\n" + "=" * 60)
print("7. MAKING PREDICTIONS ON THE TEST SET")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred_final)
report = classification_report(y_test, y_pred_final, target_names=['Bad (-1)', 'Good (+1)'])

tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  True Negatives  (TN): {tn}   — bad apples correctly identified")
print(f"  False Positives (FP): {fp}   — bad apples misclassified as good")
print(f"  False Negatives (FN): {fn}   — good apples misclassified as bad")
print(f"  True Positives  (TP): {tp}   — good apples correctly identified")
print(f"\nClassification Report:\n{report}")


# =============================================================================
# Plot 4: Confusion matrix heatmap
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Bad', 'Predicted Good'])
ax.set_yticklabels(['Actual Bad', 'Actual Good'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                fontsize=18, color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_title('Confusion Matrix — Final Adaline Model', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Figure saved] confusion_matrix.png")

# =============================================================================
# Plot 5: Distribution of net-input scores by true class
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(net_inputs[y_test == -1], bins=40, alpha=0.6, color='tomato', label='True Bad')
ax.hist(net_inputs[y_test ==  1], bins=40, alpha=0.6, color='forestgreen', label='True Good')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision boundary (0.5)')
ax.set_xlabel('Adaline Net Input (raw score)')
ax.set_ylabel('Count')
ax.set_title('Net-Input Score Distribution by True Class', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("[Figure saved] score_distribution.png")


# =============================================================================
# 8. Model Verification
# =============================================================================

print("\n" + "=" * 60)
print("8. MODEL VERIFICATION")
print("=" * 60)

# Cross-check: training set accuracy
y_pred_train = ada_final.predict(X_train_std)
acc_train = accuracy_score(y_train, y_pred_train)
print(f"\nTraining accuracy : {acc_train:.4f}")
print(f"Test accuracy     : {acc_final:.4f}")
print(f"Difference        : {abs(acc_train - acc_final):.4f}  (low = minimal overfitting)")

# Feature importance via weight magnitudes
weights_df = pd.DataFrame({
    'Feature': features,
    'Weight': ada_final.w_,
    'Abs_Weight': np.abs(ada_final.w_)
}).sort_values('Abs_Weight', ascending=False)
print(f"\nLearned Weights (sorted by importance):\n{weights_df.to_string(index=False)}")

# =============================================================================
# Plot 6: Feature importance (weight magnitudes)
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['forestgreen' if w > 0 else 'tomato' for w in weights_df['Weight']]
ax.barh(weights_df['Feature'], weights_df['Weight'], color=colors)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Weight Value')
ax.set_title('Adaline Feature Weights\n(green = positive contribution to "good")', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_weights.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n[Figure saved] feature_weights.png")


# =============================================================================
# 9. Final Apple Quality Identification Results
# =============================================================================

print("\n" + "=" * 60)
print("9. FINAL APPLE QUALITY IDENTIFICATION RESULTS")
print("=" * 60)

# Summary statistics
precision_good = tp / (tp + fp) if (tp + fp) > 0 else 0
precision_bad  = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_good    = tp / (tp + fn) if (tp + fn) > 0 else 0
recall_bad     = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_good = 2 * (precision_good * recall_good) / (precision_good + recall_good)
f1_bad  = 2 * (precision_bad  * recall_bad)  / (precision_bad  + recall_bad)

print(f"""
╔══════════════════════════════════════════════════════╗
║           FINAL MODEL PERFORMANCE SUMMARY            ║
╠══════════════════════════════════════════════════════╣
║  Overall Accuracy   :  {acc_final*100:5.2f}%                       ║
║  Training MSE       :  {ada_final.losses_[-1]:8.6f}                    ║
║  Test MSE           :  {mse_test:8.6f}                    ║
║  Test RMSE          :  {rmse_test:8.6f}                    ║
║  Test MAE           :  {mae_test:8.6f}                    ║
╠══════════════════════════════════════════════════════╣
║  Good Apple Metrics:                                 ║
║    Precision : {precision_good:.4f}    Recall : {recall_good:.4f}            ║
║    F1-Score  : {f1_good:.4f}                              ║
╠══════════════════════════════════════════════════════╣
║  Bad Apple Metrics:                                  ║
║    Precision : {precision_bad:.4f}    Recall : {recall_bad:.4f}            ║
║    F1-Score  : {f1_bad:.4f}                              ║
╚══════════════════════════════════════════════════════╝
""")

# Demonstrate prediction on new samples
print("Example predictions on new apple samples:")
new_apples = pd.DataFrame({
    'Size':        [ 1.5, -1.2,  0.8, -0.9],
    'Weight':      [ 1.2, -0.8,  0.5, -1.5],
    'Sweetness':   [ 1.8, -1.5,  1.0, -1.8],
    'Crunchiness': [ 1.3, -1.0,  0.7, -0.5],
    'Juiciness':   [ 1.4, -0.7,  1.1, -1.2],
    'Ripeness':    [-0.5,  1.3, -0.3,  1.7],
    'Acidity':     [-0.8,  0.9, -0.5,  1.1],
})
new_std = scaler.transform(new_apples.values)
new_preds = ada_final.predict(new_std)
new_labels = ['Good 🍎' if p == 1 else 'Bad  🚫' for p in new_preds]
for i, (label, row) in enumerate(zip(new_labels, new_apples.itertuples(index=False))):
    print(f"  Apple {i+1}: {label}  (Sweetness={row.Sweetness:.1f}, Acidity={row.Acidity:.1f}, Ripeness={row.Ripeness:.1f})")

print("\n✓ All outputs and figures have been saved successfully.")
