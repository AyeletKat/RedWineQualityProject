import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("winequality-red.csv")

# Separate features and target variable
X = data.drop("quality", axis=1).values
y = data["quality"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert target labels to zero-based indices
y_train = y_train - y_train.min()
y_test = y_test - y_test.min()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the neural network
class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu1 = nn.LeakyReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Fully connected layer 2
        self.relu2 = nn.LeakyReLU()  # Activation function
        self.dropout = nn.Dropout(p=0.3)  # Dropout for regularization
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # Raw logits (softmax applied during evaluation)

# Model parameters
input_size = X_train.shape[1]
hidden_size = 32  # Increased hidden size for better capacity
num_classes = len(np.unique(y))

# Initialize the model, loss function, and optimizer
model = ImprovedNN(input_size, hidden_size, num_classes)

# Calculate class weights for the imbalanced dataset
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts  # Inverse of class frequencies
weights = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)

# Optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Reduce LR every 20 epochs

# Training loop
num_epochs = 500
batch_size = 32  # Mini-batch size
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Step the scheduler
    scheduler.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    outputs = model(X_test_tensor)  # Forward pass
    _, y_pred_tensor = torch.max(outputs, 1)  # Get class predictions
    y_pred = y_pred_tensor.numpy()

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=sorted(np.unique(y)), yticklabels=sorted(np.unique(y)))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

