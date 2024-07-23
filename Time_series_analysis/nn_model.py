import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

class DNetwork(nn.Module):
    def __init__(self):
        super(DNetwork, self).__init__()
        input_size = 19
        output_size = 1
        hiden_layer_1_size = 10
        # hiden_layer_2_size = 20
        # hiden_layer_3_size = 6
        self.hidden_1 = nn.Linear(input_size, hiden_layer_1_size, bias=True)
        # self.hidden_2 = nn.Linear(hiden_layer_1_size, hiden_layer_2_size, bias=True)
        # self.hidden_3 = nn.Linear(hiden_layer_2_size, hiden_layer_3_size, bias=True)
        self.output = nn.Linear(hiden_layer_1_size, output_size, bias=True)
        self.activation_on_hidden_1 = nn.LeakyReLU(0.01)
        self.activation_on_hidden_2 = nn.LeakyReLU(0.01) # nn.Tanh()
        self.activation_on_hidden_3 = nn.LeakyReLU(0.01)
        self.activation_on_output = nn.ReLU()  # nn.Softmax()

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.activation_on_hidden_1(x)
        # x = self.hidden_2(x)
        # x = self.activation_on_hidden_2(x)
        # x = self.hidden_3(x)
        # x = self.activation_on_hidden_3(x)
        x = self.output(x)
        x = self.activation_on_output(x)
        return x

class Model:
    def __init__(self):
        self.model = DNetwork()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.00001)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray):
        inputs = torch.from_numpy(inputs).float()
        outputs = torch.from_numpy(outputs).float()
        self.optimizer.zero_grad()
        guessies = self.model(inputs)
        loss = self.criterion(guessies, outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, inputs: np.ndarray):
        return self.model(torch.from_numpy(inputs).float()).detach().numpy()







