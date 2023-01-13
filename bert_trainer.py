#%%
from bert_dataset import TextDataset
from transformers import BertModel
from torch import nn


class BERT_Classifier(nn.Module):
    def __init__(self,
                 input_size: int = 768,
                 num_classes: int = 13):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(256, 32, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(384 , 128),
                                  nn.ReLU(),
                                  nn.Linear(128, num_classes))
    def forward(self, input):
        x = self.layers(input)
        return x

if __name__ == "__main__":
    test_classifier=BERT_Classifier()

# %%
