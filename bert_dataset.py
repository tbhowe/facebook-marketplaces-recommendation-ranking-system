
#%%
import pandas as pd
from datasets import Dataset

class TextDataset():
    '''Class which defines the dataset from the csv of products in the Facebok Marketplace project

    index is currently image, to facilitate comparison with image processor. 
    However an alternative architecture is to run BERT on products CSV, repopulate with BERT outputs and then join with images.
    We can decide on this later in the development process.
    '''

    def __init__(self):
        super().__init__()
        self.load_dataframe()
        self.categories=self.text_df['labels'].unique()
        self.dataset=Dataset.from_pandas(self.text_df, preserve_index=False)
        # self.transform=transform

        # create dict of cat_name to IDX
        self.category_name_to_idx = {
            category: cat_idx for cat_idx, category in enumerate(self.categories)
        }
        
        # create dict of IDX to cat_name
        self.idx_to_category_name= {
            value: key for
            key, value in self.category_name_to_idx.items()
        }

    def load_dataframe(self):
        '''loads the products csv from the Facebook Marketplace porject into a pandas Dataframe.
        Additinoally encodes the first layer of the product category as a unique int '''
        product_df = pd.read_csv('products.csv',lineterminator='\n', index_col=0)
        product_df['price'] = product_df['price'].replace('[\£,]', '', regex=True).astype(float)
        text_df=pd.read_csv('images.csv',lineterminator='\n', index_col=0)
        self.text_df=text_df.merge(product_df, left_on ='product_id', right_on='id')
        self.text_df['cat_L1'] = [catter.split("/")[0] for catter in self.text_df['category']]
        self.text_df=pd.DataFrame().assign(features=self.text_df['product_description'], labels=self.text_df['cat_L1'])
        
    
if __name__ == "__main__":
    text_dataset=TextDataset()
    # text_dataset.text_df.head(5)
    text_dataset.dataset

# %%