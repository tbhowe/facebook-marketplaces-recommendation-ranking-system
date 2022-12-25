from Dataset import ImagesDataset
import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import os
import time
# class Module:
#     def __call__(self):

#         # other interesting stuff
#         self.forward()

#     def forward():
#         raise NotImplementedError()

class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # initialise weights and biases (parameters)
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4096, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 13),
            # torch.nn.Softmax()
        )

    def forward(self, features):
        """Takes in features and makes a prediction"""
        return self.layers(features)


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # initialise weights and biases (parameters)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            # torch.nn.Softmax()
        )

    def forward(self, features):
        """Takes in features and makes a prediction"""
        return self.layers(features)


class TransferLearning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = resnet50(weights=ResNet50_Weights)
        for param in self.layers.parameters():
            param.grad_required = False
        linear_layers = torch.nn.Sequential(
            # torch.nn.Linear(2048, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 13),
            torch.nn.Linear(2048, 13)
        )
        self.layers.fc = linear_layers
        self.initialise_weights_folders()
        # print(self.layers)

    def forward(self, x):
        return self.layers(x)

    def initialise_weights_folders(self):
        ''' method to create folder for saved weights'''
        start_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        folder_name=str('TransferLearning'+ start_time)
        if not os.path.exists('model_evaluation/' + folder_name + '/saved_weights/'):
            os.makedirs('model_evaluation/' + folder_name + '/saved_weights/') 
        self.weights_folder_name='model_evaluation/' + folder_name + '/saved_weights/'

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
