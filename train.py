from Dataset import ImagesDataset
from classifier import NeuralNetworkClassifier, CNN, TransferLearning
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms


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

    Parameters:
    - model: a pytorch model
    - dataloader: a pytorch dataloader

    Returns:
    - model: a trained pytorch model
    """

    # components of a ml algortithms
    # 1. data
    # 2. model
    # 3. criterion (loss function)
    # 4. optimiser

    writer = SummaryWriter()

    # initialise an optimiser
    optimiser = optimiser(model.parameters(), lr=lr, weight_decay=0.001)
    batch_idx = 0
    for epoch in range(epochs):  # for each epoch
        for batch in train_loader:  # for each batch in the dataloader
            features, labels = batch
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
            if batch_idx % 50 == 0:
                print('Evaluating on valiudation set')
                # evaluate the validation set performance
                val_loss, val_acc = evaluate(model, val_loader)
                writer.add_scalar("Loss/Val", val_loss, batch_idx)
                writer.add_scalar("Accuracy/Val", val_acc, batch_idx)
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

    size = 128
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop((size, size)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])

    dataset = CitiesDataset(transform=transform)
    # dataset = MNIST(root='./mnist-data', download=True, transform=transform) # TESTING
    # features, labels = dataset[0]
    # features.show()
    # print(labels)
    train_set_len = round(0.7*len(dataset))
    val_set_len = round(0.15*len(dataset))
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
    model = TransferLearning()
    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=1000,
        lr=0.0001,
        optimiser=torch.optim.AdamW
    )

