import torch
import torch.nn as nn
import torch.optim as optim


class SolarPowerForecastingNN(nn.Module):
    def __init__(self):
        super(SolarPowerForecastingNN, self).__init__()
        self.fc1 = nn.Linear(12, 21)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(21, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

    def backward_lm(self, inputs, targets, damping=0.01):
        """
        Perform one step of Levenberg-Marquardt backpropagation.
        Inputs:
            - inputs: batch of input data
            - targets: corresponding targets
            - damping: Levenberg-Marquardt damping term (lambda)
        """
        # Forward pass
        outputs = self.forward(inputs)
        loss = (outputs - targets).view(-1)

        # Compute Jacobian matrix J
        J = []
        for i in range(len(loss)):
            self.zero_grad()
            loss[i].backward(retain_graph=True)
            grad = []
            for param in self.parameters():
                grad.append(param.grad.view(-1))
            J.append(torch.cat(grad))
        J = torch.stack(J)  # Shape: (batch_size, num_params)

        # Approximate Hessian: H = J^T * J
        H = J.T @ J

        # Gradient: g = J^T * loss
        g = J.T @ loss

        # Levenberg-Marquardt update
        I = torch.eye(H.shape[0], device=H.device)
        H_lm = H + damping * I
        delta = torch.linalg.solve(H_lm, -g)

        # Update parameters manually
        idx = 0
        for param in self.parameters():
            num_params = param.numel()
            param_update = delta[idx:idx+num_params].view(param.size())
            param.data += param_update
            idx += num_params

