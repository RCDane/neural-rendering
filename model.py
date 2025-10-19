import torch



class TextureEncoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_layer_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_layer_dim)
        self.fc2 = torch.nn.Linear(hidden_layer_dim, output_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class SimpleNeuralBSDF(torch.nn.Module):
    def __init__(self, input_layer_dim, hidden_layer_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_layer_dim, hidden_layer_dim)
        # self.bn1 = torch.nn.BatchNorm1d(hidden_layer_dim)
        self.fc2 = torch.nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = torch.nn.Linear(hidden_layer_dim, output_dim)
        self.activation = torch.nn.ReLU()
        # self.output_activation = torch.nn.Sigmoid()  # Assuming BSDF values are in [0, 1]   

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x
