import torch
import torch.nn as nn
import torch.optim as optim

# Define a basic regression model (single stage)
class SimpleRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

# Define cascaded regression model (multiple stages)
class CascadedRegressor(nn.Module):
    def __init__(self, input_size, output_size, num_stages=3):
        super(CascadedRegressor, self).__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([SimpleRegressor(input_size, output_size) for _ in range(num_stages)])
    
    def forward(self, x):
        # Initial guess (zero prediction)
        pred = torch.zeros(x.size(0), output_size).to(x.device)
        
        for i in range(self.num_stages):
            # Update prediction using current stage
            correction = self.stages[i](x)
            pred = pred + correction  # Update prediction with correction
            
        return pred

# Example usage
input_size = 10  # Assume 10 input features
output_size = 2  # For example, predicting 2D coordinates

# Create cascaded regressor model
model = CascadedRegressor(input_size, output_size, num_stages=3)

# Create dummy input data
x = torch.randn(32, input_size)  # 32 samples, each with 10 features
y_true = torch.randn(32, output_size)  # 32 true targets, each with 2 output values

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop (simple example)
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss = criterion(y_pred, y_true)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
