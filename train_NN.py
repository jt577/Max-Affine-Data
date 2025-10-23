import numpy as np
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Create the directory for snapshots
snapshot_dir = '3layer_wd_1e-8_0_25eV'
os.makedirs(snapshot_dir, exist_ok=True)

# Load the list from the file
with open('OH_elements.json', 'r') as file:
    OH_elements = json.load(file)

# Load the list from the file
with open('formation_energies_valid_0_25eV.json', 'r') as file:
    formation_energies_valid = json.load(file)

# Load the dictionary from the file
with open('p_dict_oh.json', 'r') as file:
    p_dict_oh = json.load(file)

# Load features tensor list
features_tensor = torch.load('features_tensor_0_25eV.pt')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Convert the list of tensors to a single tensor
features_tensor_combined = torch.stack(features_tensor)

# Normalize the target values
scaler_y = MinMaxScaler(feature_range=(0, 1))
formation_energies_tensor = torch.tensor(scaler_y.fit_transform(np.array(formation_energies_valid).reshape(-1, 1)), dtype=torch.float32).squeeze()

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(features_tensor_combined, formation_energies_tensor, test_size=0.2, random_state=23)

# Convert numpy arrays to PyTorch tensors and transfer them to the GPU
X_train = torch.tensor(X_train).to(device)
X_test = torch.tensor(X_test).to(device)
y_train = torch.tensor(y_train).to(device)
y_test = torch.tensor(y_test).to(device)

# Ensure y_train and y_test are reshaped to match the prediction shape
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

# Create DataLoader for mini-batch training
batch_size = 512
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSet, self).__init__()
        self.rho = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Adjust the input dimension to match the concatenated features
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # Feed the concatenated tensor into rho
        output = self.rho(x)
        return output

# Parameters
input_dim = features_tensor[0].shape[-1]  # Dimension of each feature vector
hidden_dim = 512  # Hidden layer size
output_dim = 1  # Single value output (formation energy)

# Initialize the model and transfer it to the GPU
model = DeepSet(input_dim, hidden_dim, output_dim).to(device)

# Check if the model file exists and load it if it does
model_path = 'model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from", model_path)
else:
    print("No pre-existing model found. Starting training from scratch.")

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

# Training the neural network
num_epochs = 1200
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Transfer mini-batch data to the GPU
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
    
    if (epoch+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            X_train, X_test = X_train.to(device), X_test.to(device)
            y_train, y_test = y_train.to(device), y_test.to(device)
            
            predicted_values_train = model(X_train)
            predicted_values_test = model(X_test)
            
            y_train_inverse = scaler_y.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
            predicted_values_train_inverse = scaler_y.inverse_transform(predicted_values_train.cpu().numpy().reshape(-1, 1))
            y_test_inverse = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
            predicted_values_test_inverse = scaler_y.inverse_transform(predicted_values_test.cpu().numpy().reshape(-1, 1))
        
        # Save plot for training data
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train_inverse, predicted_values_train_inverse, color='blue', edgecolor='k', alpha=0.7, label='Training Data')
        plt.plot([min(y_train_inverse), max(y_train_inverse)], [min(y_train_inverse), max(y_train_inverse)], color='red', linestyle='--', linewidth=2, label='Ideal Fit')
        plt.xlabel('Actual Formation Energies (eV)', fontsize=16)
        plt.ylabel('Predicted Formation Energies (eV)', fontsize=16)
        plt.legend()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.savefig(f'{snapshot_dir}/training_epoch_{epoch+1}.png')
        plt.close()

# After training the model
torch.save(model.state_dict(), 'model_tanh_wd_1e-8_3layer_0_25eV_run1.pth')

# Evaluating the model
model.eval()
with torch.no_grad():
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)
    
    predicted_values_train = model(X_train)
    predicted_values_test = model(X_test)
    
    r_squared_train = 1 - torch.sum((y_train - predicted_values_train) ** 2) / torch.sum((y_train - torch.mean(y_train)) ** 2)
    r_squared_test = 1 - torch.sum((y_test - predicted_values_test) ** 2) / torch.sum((y_test - torch.mean(y_test)) ** 2)

print("R-squared (Training):", r_squared_train.item())
print("R-squared (Testing):", r_squared_test.item())

# Inverse transform the normalized predictions and target values
y_train_inverse = scaler_y.inverse_transform(y_train.cpu().numpy().reshape(-1, 1))
predicted_values_train_inverse = scaler_y.inverse_transform(predicted_values_train.cpu().numpy().reshape(-1, 1))
y_test_inverse = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
predicted_values_test_inverse = scaler_y.inverse_transform(predicted_values_test.cpu().numpy().reshape(-1, 1))

# Extract scaler parameters
scaler_params = {
    'min_': scaler_y.min_.tolist(),
    'scale_': scaler_y.scale_.tolist(),
    'data_min_': scaler_y.data_min_.tolist(),
    'data_max_': scaler_y.data_max_.tolist(),
    'data_range_': scaler_y.data_range_.tolist(),
    'feature_range': scaler_y.feature_range
}

# Save the scaler parameters as JSON
with open('scaler_y_0_25eV.json', 'w') as f:
    json.dump(scaler_params, f)

# Calculate MAE
mae_train = mean_absolute_error(y_train_inverse, predicted_values_train_inverse)
mae_test = mean_absolute_error(y_test_inverse, predicted_values_test_inverse)
print("MAE (Training):", mae_train)
print("MAE (Testing):", mae_test)

# Calculate RMSE
rmse_train = np.sqrt(mean_squared_error(y_train_inverse, predicted_values_train_inverse))
rmse_test = np.sqrt(mean_squared_error(y_test_inverse, predicted_values_test_inverse))
print("RMSE (Training):", rmse_train)
print("RMSE (Testing):", rmse_test)

# Plotting the actual vs predicted energies for training data
plt.figure(figsize=(8, 6))
plt.scatter(y_train_inverse, predicted_values_train_inverse, color='blue', edgecolor='k', alpha=0.7, label='Training Data')
plt.plot([min(y_train_inverse), max(y_train_inverse)], [min(y_train_inverse), max(y_train_inverse)], color='red', linestyle='--', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Formation Energies (eV)', fontsize=16)
plt.ylabel('Predicted Formation Energies (eV)', fontsize=16)
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

# Add RMSE, MAE, and R-squared to the plot for training data
plt.text(0.75, 0.05, f'RMSE: {rmse_train:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.text(0.75, 0.10, f'MAE: {mae_train:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.text(0.75, 0.15, f'R^2: {r_squared_train.item():.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.savefig(f'{snapshot_dir}/final_training_plot.png')
plt.close()

# Plotting the actual vs predicted energies for testing data
plt.figure(figsize=(8, 6))
plt.scatter(y_test_inverse, predicted_values_test_inverse, color='green', edgecolor='k', alpha=0.7, label='Testing Data')
plt.plot([min(y_test_inverse), max(y_test_inverse)], [min(y_test_inverse), max(y_test_inverse)], color='red', linestyle='--', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Formation Energies (eV)', fontsize=16)
plt.ylabel('Predicted Formation Energies (eV)', fontsize=16)
plt.legend()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)

# Add RMSE, MAE, and R-squared to the plot for testing data
plt.text(0.75, 0.05, f'RMSE: {rmse_test:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.text(0.75, 0.10, f'MAE: {mae_test:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.text(0.75, 0.15, f'R^2: {r_squared_test.item():.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
plt.savefig(f'{snapshot_dir}/final_testing_plot.png')
plt.close()
