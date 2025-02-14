import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Add species column (target labels)

# Define features (X) and target labels (y)
X = iris.data[:, :2]  # Only use the first two features: Sepal Length, Sepal Width
y = iris.target  # Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for Logistic Regression and SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
log_reg_model = LogisticRegression(max_iter=200)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear')

# Train models
log_reg_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

# Print accuracy and classification report for each model
print("===== Logistic Regression Model =====")
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Accuracy: {accuracy_log_reg:.2f}")
print(classification_report(y_test, y_pred_log_reg, target_names=iris.target_names))

print("\n===== Random Forest Model =====")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf:.2f}")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

print("\n===== SVM Model =====")
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy: {accuracy_svm:.2f}")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

# Comparison of model accuracy and performance
print("\n===== Model Comparison =====")
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.2f}")
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print(f"SVM Accuracy: {accuracy_svm:.2f}")

# Plotting function for decision boundaries
def plot_decision_boundary(model, X, y, ax):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
    ax.set_title(f'{model.__class__.__name__}')
    return scatter

# Plot the decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Logistic Regression
plot_decision_boundary(log_reg_model, X, y, axes[0])
axes[0].set_xlabel("Sepal Length")
axes[0].set_ylabel("Sepal Width")

# Random Forest
plot_decision_boundary(rf_model, X, y, axes[1])
axes[1].set_xlabel("Sepal Length")
axes[1].set_ylabel("Sepal Width")

# SVM
plot_decision_boundary(svm_model, X, y, axes[2])
axes[2].set_xlabel("Sepal Length")
axes[2].set_ylabel("Sepal Width")

# Add a color bar
fig.colorbar(axes[0].collections[0], ax=axes, orientation='horizontal')

plt.show()
