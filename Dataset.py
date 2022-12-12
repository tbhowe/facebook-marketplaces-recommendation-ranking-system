from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os



class TestDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.load_dataframe()
        self.categories=self.image_df['cat_L1'].unique()
        self.all_images=self.image_df['id_x']
        # create dict of cat_name to IDX
        self.category_name_to_idx = {
            category: cat_idx for cat_idx, category in enumerate(self.categories)
        }
        # create dict of IDX to cat name
        self.idx_to_category_name= {
            value: key for
            key, value in self.category_name_to_idx.items()
        }

    def __getitem__(self, idx):
       return self.get_X_y_from_img_idx(idx)

    def __repr__(self):
        return "hello"  # str(self.cities)
    
    def __len__(self):
        return len(self.all_images)
        # return len(self.all_imgs)
    def load_dataframe(self):
        product_df = pd.read_csv('products.csv',lineterminator='\n', index_col=0)
        product_df['price'] = product_df['price'].replace('[\£,]', '', regex=True).astype(float)
        image_df=pd.read_csv('images.csv',lineterminator='\n', index_col=0)
        self.image_df=image_df.merge(product_df, left_on ='product_id', right_on='id')
        self.image_df['cat_L1'] = [catter.split("/")[0] for catter in self.image_df['category']]
    
    def get_X_y_from_img_idx(self, idx):
        cwd = os.getcwd()
        image_ID=self.image_df.iloc[idx]['id_x']
        image_fp = ( cwd + '/cleaned_images/' + image_ID + '.jpg')
        # print(img_fp)
        img = Image.open(image_fp)
        # if self.transform:
        #     img = self.transform(img)
        category_idx = self.category_name_to_idx[self.image_df.iloc[idx]['cat_L1']]
        return img, category_idx

my_dataset=TestDataset()
my_dataset[10]