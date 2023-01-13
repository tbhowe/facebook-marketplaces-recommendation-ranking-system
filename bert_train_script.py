#%%
from bert_dataset import TextDataset
from bert_trainer import BERT_Classifier
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms
from torch.optim import lr_scheduler

torch.manual_seed=123

def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    lr=0.001,
    epochs=20,
    optimiser=torch.optim.Adam
    ):
    '''train loop for 1d Conv of BERT outputs, with lr scheduling'''
    writer = SummaryWriter()

    # initialise an optimiser
    optimiser = optimiser(model.parameters(), lr=lr)  # weight_decay=0.001
    scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=[5,10], gamma=0.1,verbose=True)
    batch_idx = 0
    epoch_idx= 0
    for epoch in range(epochs):  # for each epoch
        # 
        
        print('Epoch:', epoch_idx,'LR:', scheduler.get_lr())
        weights_filename=model.weights_folder_name + '_latest_weights.pt'
        epoch_idx +=1
        torch.save(model.state_dict(), weights_filename)

        for batch in train_loader:  # for each batch in the dataloader
            features, labels = batch
            # print(size(images))
            prediction = model(features)  # make a prediction
            # compare the prediction to the label to calculate the loss (how bad is the model)
            loss = F.cross_entropy(prediction, labels)
            loss.backward()  # calculate the gradient of the loss with respect to each model parameter
            optimiser.step()  # use the optimiser to update the model parameters using those gradients
            print("Epoch:", epoch, "Batch:", batch_idx,
                  "Loss:", loss.item())  # log the loss
            optimiser.zero_grad()  # zero grad
            writer.add_scalar("Loss/Train", loss.item(), batch_idx)
            batch_idx += 1
            
        print('Evaluating on valiudation set')
        # evaluate the validation set performance
        val_loss, val_acc = evaluate(model, val_loader)
        writer.add_scalar("Loss/Val", val_loss, batch_idx)
        writer.add_scalar("Accuracy/Val", val_acc, batch_idx)
        scheduler.step()
    # evaluate the final test set performance
    
    print('Evaluating on test set')
    test_loss = evaluate(model, test_loader)
    # writer.add_scalar("Loss/Test", test_loss, batch_idx)
    model.test_loss = test_loss
    
    return model   # return trained model
    

def evaluate(model, dataloader):
    losses = []
    correct = 0
    n_examples = 0
    for batch in dataloader:
        features, labels = batch
        prediction = model(features)
        loss = F.cross_entropy(prediction, labels)
        losses.append(loss.detach())
        correct += torch.sum(torch.argmax(prediction, dim=1) == labels)
        n_examples += len(labels)
    avg_loss = np.mean(losses)
    accuracy = correct / n_examples
    print("Loss:", avg_loss, "Accuracy:", accuracy.detach().numpy())
    return avg_loss, accuracy


if __name__ == "__main__":

    dataset = TextDataset()
    train_set_len = round(0.8*len(dataset))
    val_set_len = round(0.1*len(dataset))
    test_set_len = len(dataset) - val_set_len - train_set_len
    split_lengths = [train_set_len, val_set_len, test_set_len]
    # split the data to get validation and test sets
    train_set, val_set, test_set = random_split(dataset, split_lengths)

    batch_size = 16
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    # nn = NeuralNetworkClassifier()
    # cnn = CNN()
    model = BERT_Classifier()
    
    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=20,
        lr=0.001,
        optimiser=torch.optim.AdamW
        )



# %%
