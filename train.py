#%%
from Classifier import CNN
from Dataset import ImagesDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import random_split


# def overtrain_on_one_batch()

def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    lr=0.1,
    epochs=100,
    optimiser=torch.optim.SGD
    ):
    """
        Trains a neural network on a dataset and returns the trained model

        Inputs:
        - model: a pytorch model
        - dataloader: a pytorch dataloader

        Returns:
        - model: a trained pytorch model
        """
    # initialise tensorboard input
    writer = SummaryWriter()

    # initialise an optimiser
    optimiser = optimiser(model.parameters(), lr=lr)

    global_idx = 0

    # line below ovetrains on one example
    features,labels=next(iter(train_loader))

    for epoch in range(epochs):  

        for batch in train_loader:  
            # features, labels = batch -commenting out to pass one example for overtrain

            # make prediction
            prediction = model(features)  
            
            # loss calculation and propagation
            loss = F.cross_entropy(prediction, labels)
            loss.backward()  # calculate loss gradients
            optimiser.step()  # update model params from gradients

            print("Epoch:", epoch, "Batch:", global_idx,
                  "Loss:", loss.item())  # log the loss
            optimiser.zero_grad()  # zero grad
            writer.add_scalar("Loss/Train", loss.item(), global_idx)
            global_idx += 1

            if global_idx % 20 == 0:
                print('Evaluating on valiudation set')
                val_loss, val_acc = evaluate(model, val_loader)
                writer.add_scalar("Loss/Val", val_loss, global_idx)
                writer.add_scalar("Accuracy/Val", val_acc, global_idx)

    # evaluate the final test set performance
    print('Evaluating on test set')
    test_loss = evaluate(model, test_loader)
    writer.add_scalar("Loss/Test", test_loss, global_idx)
    model.test_loss = test_loss
    return model   # return trained model

def evaluate(model, dataloader):
    '''evaluates the performance of the model at current state'''
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
    print("Loss =", avg_loss, "Accuracy =", accuracy.detach().numpy())
    return avg_loss, accuracy

# Main code:

# Initialise dataset
FacebookImagesDataset=ImagesDataset()

device = torch.device("cpu")

# Create train-test split
train_set_len = round(0.7*len(FacebookImagesDataset))
val_set_len = round(0.15*len(FacebookImagesDataset))
test_set_len = len(FacebookImagesDataset) - val_set_len - train_set_len
split_lengths = [train_set_len, val_set_len, test_set_len]
train_set, val_set, test_set = random_split(FacebookImagesDataset, split_lengths)

# Initialise data loaders and model
batch_size=16
train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)
model = CNN()


# Train the model
train(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs=1000,
    lr=0.1,
    optimiser=torch.optim.SGD
    )

# %%
