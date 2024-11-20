import torch
import torch.nn as nn
import torch.optim as optim

# Define the Perceptron model for the NOT gate
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input for NOT gate

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))  # Sigmoid activation for binary output
        return out

# Initialize the model, loss function, and optimizer
model = Perceptron()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent optimizer

# Training data for NOT gate
data = torch.tensor([[0.0], [1.0]])  # Inputs
labels = torch.tensor([[1.0], [0.0]])  # Expected outputs for NOT gate

# Training the Perceptron
epochs = 1000
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the trained model on NOT gate inputs
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_output = model(data)
    predicted = test_output.round()  # Round to get binary output
    print(f'Predicted outputs for NOT gate:\n{predicted}')
