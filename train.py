from Dataset import ImagesDataset
from Classifier import TransferLearning
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import random_split
from torchvision import transforms
from torch.optim import lr_scheduler

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
    scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=[5,20,50], gamma=0.1,verbose=True)
    state_dict=torch.load( 'model_evaluation/TransferLearning2022-12-25-08:15:53/saved_weights/_latest_weights.pt' )
    model.load_state_dict(state_dict)
    batch_idx = 0
    epoch_idx= 0
    for epoch in range(epochs):  # for each epoch
        # 
        
        print('Epoch:', epoch_idx,'LR:', scheduler.get_lr())
        weights_filename=model.weights_folder_name + '_' + str(epoch) + '_latest_weights.pt'
        epoch_idx +=1
        torch.save(model.state_dict(), weights_filename)
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
            if batch_idx % 100 == 0:
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

    size = 64
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop((size,size), pad_if_needed=True),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    dataset = ImagesDataset(transform=transform)
    train_set_len = round(0.7*len(dataset))
    val_set_len = round(0.15*len(dataset))
    test_set_len = len(dataset) - val_set_len - train_set_len
    split_lengths = [train_set_len, val_set_len, test_set_len]
    # split the data to get validation and test sets
    train_set, val_set, test_set = random_split(dataset, split_lengths)

    batch_size = 32
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
        epochs=20,
        lr=0.0001,
        optimiser=torch.optim.AdamW
        
    )
