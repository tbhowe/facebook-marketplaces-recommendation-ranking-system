from Dataset import ImagesDataset
import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import os
import time

class TransferLearning(torch.nn.Module):
    '''Creates an instance of ReseNet50, populates its weights with the trained weights, and adds a single linear layer
    as an output layer
    '''
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

            # torch.nn.Linear(2048, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 13),
        )
        self.layers.fc = linear_layers
        self.initialise_weights_folders()
        self.image_size=64
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomCrop((self.image_size), pad_if_needed=True),
            transforms.ToTensor(),
            ])
        

    def forward(self, x):
        '''defines the forward pass for the model'''
        return self.layers(x)

    def initialise_weights_folders(self):
        ''' method to create folder for saved weights'''
        start_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        folder_name=str('TransferLearning'+ start_time)
        if not os.path.exists('model_evaluation/' + folder_name + '/saved_weights/'):
            os.makedirs('model_evaluation/' + folder_name + '/saved_weights/') 
        self.weights_folder_name='model_evaluation/' + folder_name + '/saved_weights/'

   
    def transform_image(self,img):
        '''method to transform an input image to tensor and give correct dimensionality for input as prediction'''
        img=self.transform(img).unsqueeze(0)
        
        