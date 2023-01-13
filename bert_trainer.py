#%%
from bert_dataset import TextDataset
from torch import nn
import time
import os


class BERT_Classifier(nn.Module):
    ''' 1d convolutional network, taking in BERT model last hidden layer as input'''
    def __init__(self,
                 input_size: int = 768,
                 num_classes: int = 13):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(192 , num_classes)
                                  )
                                 

        self.initialise_weights_folders()

    def forward(self, input):
        x = self.layers(input)
        return x
    
    def initialise_weights_folders(self):
        ''' method to create folder for saved weights'''
        start_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        folder_name=str('BERT_1d_conv'+ start_time)
        if not os.path.exists('model_evaluation/' + folder_name + '/saved_weights/'):
            os.makedirs('model_evaluation/' + folder_name + '/saved_weights/') 
        self.weights_folder_name='model_evaluation/' + folder_name + '/saved_weights/'

if __name__ == "__main__":
    test_classifier=BERT_Classifier()
    params=test_classifier.parameters()

# %%
