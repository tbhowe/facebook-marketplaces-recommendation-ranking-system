# Facebook Marketplace Recommendation Ranking System

 ## Milestone 1-3: data cleaning

 The files clean_image_data and clean_tabular data contain code to clean the marketplace data. 

 In the case of the tabular data, the code loads the data into a dataframe, removes the £ sign from the price column, and re-casts the series as float data. The resulting modified dataframe is then saved to a new .csv file.

## Milestone 4: create and use the vision model

the Pytorch library is used to create a transfer learning model, utilising the ResNet50 model with a single Linear layer to convert its output to the correct number of categories. More complex architectures were considered during training, but were abandoned due to compute constraints using the large dataset.

### Dataset.py

The file Dataset.py describes a class, ImagesDataset, which inherits from torch.utils.data.Dataset. 
The following modifications were made to facilitate the use of our dataset from the Facebook Marketing project:

 - __getitem__() and __len__() methods were modified to apply to our data. The __init__ method contains an encoder and decoder dictionary, which can be called to convert between label names and their int indices. The __len__ method returns the number of images in the dataset.

 - creaiton of a load_dataframe() method - this loads the products csv from the Facebook Marketplace porject into a pandas dataframe, merged with a dataframe of the associated image files. This creates a new dataframe of n_images * m columns, with the column values filled from the corresponding entry in products.csv.  It adds two new columns to the original dataframe: one column for the top-level of the product category tree, which form our category labels, and one for those category labels cast to an int index.

 - get_X_y_from_img_id() method - when called by the __getitem__() method, this method takes in a dataframe row index and returns a PIL image (features) and an int value (categorical label).

 -show_example_image() method - takes a dataframe row index and shows the associated image.

 - get_value_frequencies() method - when called, this method generates a dict of the relative frequencies of each category label in the dataset, expressed as a percentage. This is useful when checking that the dataset is balanced. It is also used during training experiments on my M1 Macbook, as the classifier is very slow to train on the full dataset, and brief experiments can utilise a random subset of the dataset. This method allows to check for consistency in category frequencies between the subsample and the full dataset.

 ## Classifier.py







 