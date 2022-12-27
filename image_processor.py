#%%
from Classifier import TransferLearning
from Dataset import ImagesDataset
import torch
from PIL import Image
from torchvision import transforms

def get_prediction(img):
    '''function to generate a prediction based on an input image'''
    img=transform(img).unsqueeze(0)
    assert torch.is_tensor(img)
    prediction = model.forward(img) 
    probs = torch.nn.functional.softmax(prediction, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), dataset.idx_to_category_name[classes.item()]

# Initialise model and dataset (the latter just to get a test image)
model = TransferLearning()
transform=model.transform
dataset=ImagesDataset(transform) # This line should be unneccessary once we pass a user's image instead


state_dict=torch.load('final_models/image_model.pt ')
model.load_state_dict(state_dict)
model.eval()
image_fp = ('cleaned_images/0c2f81f8-7d98-42e2-9d7d-836335fa08df.jpg')
example_image=Image.open(image_fp)
get_prediction(example_image)


# %%
