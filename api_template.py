import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from Classifier import TransferLearning

class ImageClassifier(TransferLearning):
    '''an image classifier baseds on the TransferLearning class in Classifier.py'''
    def __init__(self,
                 decoder: dict = None):
        super().__init__()
        with open('idx_to_cat.pickle', 'rb') as handle:
            self.decoder = pickle.load(handle)
        
    def forward(self, x):
        '''defines the forward pass for the model'''
        return self.layers(x)

    def predict(self, image):
        '''takes in an image and returns predicted item class and confidence score'''
        with torch.no_grad():
            prediction = self.forward(image)
            probs = torch.nn.functional.softmax(prediction, dim=1)
            conf, classes = torch.max(probs, 1)
            return conf.item(), self.decoder[classes.item()], probs

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str

try:
##############################################################
# TODO                                                       #
# Load the image model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the image model   #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# image_decoder.pkl                                          #
##############################################################
    image_model=ImageClassifier()
    state_dict=torch.load('final_models/image_model.pt ')
    image_model.load_state_dict(state_dict=state_dict)
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")




try:
##############################################################
# TODO                                                       #
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################

    class ImageProcessor:
        '''a class that applies the transforms necessary to preprocess an image for passing to the model '''
        def __init__(self):
            '''class constructor - inputs are dicts to transform indices to categories'''
            self.transform=image_model.transform
        
        def transform_image(self,img):
            '''method to transform an input image to tensor and give correct dimensionality for input as prediction'''
            return self.transform(img).unsqueeze(0)
    
    image_processor=ImageProcessor()
    pass
  
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/image')
##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    image_tensor=image_processor.transform_image(pil_image)
    confidence,category, probabilities = image_model.predict(image_tensor)
    return JSONResponse(content={
    "category": category, 
    "confidence": confidence,
    "probabilites" :  probabilities
        })
 
    
if __name__ == '__main__':
#   uvicorn.run("api:app", host="0.0.0.0", port=8080)
    uvicorn.run("api:app", host="127.0.0.1", port=8080)