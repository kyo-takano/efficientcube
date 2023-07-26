import torch
from torch import nn

class LinearBlock(nn.Module):
    """
    Linear layer with ReLU and BatchNorm
    """
    def __init__(self, input_prev, embed_dim):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_prev, embed_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, inputs):
        x = inputs
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with two linear layers
    """
    def __init__(self, embed_dim):
        super(ResidualBlock, self).__init__()
        self.linearblock_1 = LinearBlock(embed_dim, embed_dim)
        self.linearblock_2 = LinearBlock(embed_dim, embed_dim)

    def forward(self, inputs):
        x = inputs
        x = self.linearblock_1(x)
        x = self.linearblock_2(x)
        x += inputs # skip-connection
        return x

class Model(nn.Module):
    """
    Fixed architecture following DeepCubeA.
    """
    def __init__(self, input_dim=324, output_dim=12):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.Stack = nn.Sequential(
            LinearBlock(input_dim, 5000),
            LinearBlock(5000,1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000)
        )
        self.Prediction = nn.Linear(1000, output_dim)

    def forward(self, inputs):
        # int indices => float one-hot vectors
        x = nn.functional.one_hot(inputs, num_classes=6).to(torch.float)
        x = x.reshape(-1, self.input_dim)
        x = self.Stack(x)
        logits = self.Prediction(x)
        return logits

class ScalableModel(nn.Module):
    """
    Scalable MLP (without residual connections) as in Section 7 Scaling Law (pp. 9-10).
    """
    def __init__(self, embed_dim, num_hidden_layers, input_dim=324, output_dim=12):
        super(ScalableModel, self).__init__()
        self.input_dim = input_dim
        layers = [
            LinearBlock(embed_dim, embed_dim) if i else LinearBlock(input_dim, embed_dim)
            for i in range(num_hidden_layers)
        ]
        self.Stack = nn.Sequential(*layers)
        self.Prediction = nn.Linear(embed_dim, output_dim)

    def forward(self, inputs):
        # int indices => float one-hot vectors
        x = nn.functional.one_hot(inputs, num_classes=6).to(torch.float)
        x = x.reshape(-1, self.input_dim)
        x = self.Stack(x)
        logits = self.Prediction(x)
        return logits

if __name__=="__main__":
    # Define `model` and load it on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

    # Test prediction with a fake input tensor
    model.eval()
    sample = torch.randint(0, 6, (54, )).to(device)
    print(f"{sample.shape=}, {sample=}")

    with torch.no_grad():
        logits = model(sample)[0, :]
        pdist = nn.functional.softmax(logits, dim=-1)
    print(f"{pdist.shape=}, {pdist=}")

