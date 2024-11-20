import torch
import torch.nn as nn
import torch.optim as optim

# Define the Perceptron model for the OR gate
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Single-layer perceptron

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))  # Sigmoid activation for binary output
        return out

# Initialize the model, loss function, and optimizer
input_size = 2  # OR gate has two input features
model = Perceptron(input_size)
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent optimizer

# Training data for OR gate
data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # Inputs
labels = torch.tensor([[0.0], [1.0], [1.0], [1.0]])  # Expected outputs

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

# Testing the trained model on OR gate inputs
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_output = model(data)
    predicted = test_output.round()  # Round to get binary output
    print(f'Predicted outputs for OR gate:\n{predicted}')
