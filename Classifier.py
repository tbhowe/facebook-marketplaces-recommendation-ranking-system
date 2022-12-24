
import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import os
import time


class TransferLearning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = resnet50(weights=ResNet50_Weights)
        for param in self.layers.parameters():
            param.grad_required = False
        linear_layers = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 13),
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
    
  




if __name__ == "__main__":
    model = TransferLearning()
    prediction = model(features)
    print('Prediction:', prediction)
    print('Label:', label)
