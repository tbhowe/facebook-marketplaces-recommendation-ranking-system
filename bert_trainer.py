#%%
from bert_dataset import TextDataset
import torch
from transformers import BertModel

class BERT_classifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        for param in self.layers.parameters():
            param.grad_required = False     

        # for param in self.layers.layer4.parameters():  
        #     param.grad_required = True
        #     param.lr=0.000001
        linear_layers = torch.nn.Sequential(
            torch.nn.Linear(38400, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 13)
        )
        self.layers.fc = linear_layers
        # print(self.layers)

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    test_classifier=BERT_classifier()

# %%
