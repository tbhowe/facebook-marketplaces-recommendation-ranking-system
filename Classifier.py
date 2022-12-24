from Dataset import ImagesDataset
import torch
from torchvision.models import resnet50


class TransferLearning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = resnet50()
        for param in self.layers.parameters():
            param.grad_required = False
        linear_layers = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 13),
        )
        self.layers.fc = linear_layers
        # print(self.layers)

    def forward(self, x):
        return self.layers(x)

# model = TransferLearning()
# optimiser = torch.optim.Adam(model.feature_extractor.parameters(), lr=0.00001)
# # do trainign
# optimiser.load_state_dict['lr']


if __name__ == "__main__":
    # citiesDataset = CitiesDataset()
    # example = citiesDataset[0]
    # print(example)
    # features, label = example
    # nn = NeuralNetworkClassifier()
    model = TransferLearning()
    prediction = model(features)
    print('Prediction:', prediction)
    print('Label:', label)
