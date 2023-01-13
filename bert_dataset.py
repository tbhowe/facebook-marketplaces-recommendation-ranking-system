
#%%
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer
from transformers import BertModel
from transformers import DataCollatorForTokenClassification
import torch

class TextDataset(Dataset):
    '''Class which defines the dataset from the csv of products in the Facebok Marketplace project
    '''

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.text_df=self.load_dataframe()
        self.categories=self.text_df['labels'].unique()
        self.encoder = {y: x for (x, y) in enumerate(set(self.categories))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.categories))}
        self.max_length = 50

    def load_dataframe(self):
        '''loads the products csv from the Facebook Marketplace porject into a pandas Dataframe.
        Additinoally encodes the first layer of the product category as a unique int '''
        product_df = pd.read_csv('products.csv',lineterminator='\n', index_col=0)
        product_df['price'] = product_df['price'].replace('[\£,]', '', regex=True).astype(float)
        text_df=pd.read_csv('images.csv',lineterminator='\n', index_col=0)
        self.text_df=text_df.merge(product_df, left_on ='product_id', right_on='id')
        self.text_df['cat_L1'] = [catter.split("/")[0] for catter in self.text_df['category']]
        return pd.DataFrame().assign(description=self.text_df['product_description'], labels=self.text_df['cat_L1'])
    
    def __getitem__(self, index):
        label = self.text_df['labels'][index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        description =self.text_df['description'][index]
        features=self.tokenizer([description], max_length=self.max_length, padding='max_length', truncation=True)
        features = {key:torch.LongTensor(value) for key, value in features.items()}

        return features, label

    def __len__(self):
        return self.text_df.shape[0]


    # def tokenise_function(self,example):
    #     return self.tokenizer(example['description'], truncation=True)
        

        
    
if __name__ == "__main__":
    text_dataset=TextDataset()
    test_features, test_label=text_dataset[1]
    # print(test_features)
    # print(test_label)
    
    model=BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model.eval()
    test_out=model(**test_features).last_hidden_state
    print(test_out.logits.size())
    # print(len(text_dataset))
# %%
