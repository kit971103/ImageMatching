from itertools import pairwise
from tkinter import X
import torch
import torch.nn.functional as F
from torch import nn, optim

class Network(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_features = [] , drop_out = 0.2) -> None:
        ''' Builds a feedforward network with arbitrary hidden layers(at least one).
            
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers(cant be empty)
            
        '''
        if not hidden_layers_features: 
            raise NotImplementedError("hidden_layers_features cant be empty")

        super().__init__()

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_features = a, out_features = b) for a, b in pairwise( [in_features] + hidden_layers_features)
        ])
        self.fc_out = nn.Linear(in_features = hidden_layers_features[-1], out_features = out_features)
        
        self.drop_out = nn.Dropout(p = drop_out)
    
    def forward(self, input):

        output = input.view(input.shape[0], -1)
        
        for hidden_layer in self.hidden_layers:
            output = self.drop_out(F.relu(hidden_layer(output)))
        
        output = F.log_softmax(self.fc_out(output), dim = 1)

        return output

def train_cycle(model, criterion, optimizer,  trainloader, testloader, num_ecpochs = 5, print_every = 100):

    trainnig_losses, validation_losses, accuracy_tracking = [], [], []

    for epoch_id in range(num_ecpochs):
        
        model.train()
        loss = train(model, criterion, optimizer, trainloader, print_every = print_every)
        trainnig_losses.append(loss)
        
        model.eval()
        loss, accuracy = validate(model, criterion, testloader, print_every = print_every)
        validation_losses.append(loss)
        accuracy_tracking.append(accuracy)

        print(f"{epoch_id+1:>2}/{num_ecpochs:>2}\ttrain_loss: {trainnig_losses[-1]:.4f}\ttest_loss: {validation_losses[-1]:.4f}\taccuracy: {accuracy_tracking[-1]*100:.4f}%")
    return trainnig_losses, validation_losses, accuracy_tracking

def train(model, criterion, optimizer, trainloader, print_every = 100):
    
    acc_losses = 0 
    for iter_id, (images, labels) in enumerate(trainloader, 1):

        logit = model(images)
        loss = criterion(logit, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_losses+=loss.item()
        
        if iter_id % print_every == 0:
            print(f"\t{iter_id}/{len(trainloader)}\ttrain_loss: {acc_losses/iter_id:.4f}")

    return acc_losses/len(trainloader)

def validate(model, criterion, testloader, print_every = 100):
    
    with torch.no_grad():
        acc_losses = 0 
        accuracy = 0
        for iter_id, (images, labels) in enumerate(testloader, 1):

            logit = model(images)
            loss = criterion(logit, labels)
            acc_losses+=loss.item()

            _, top_class = logit.topk(1, dim = 1)
            equals = labels == top_class.view(*labels.shape)
            accuracy += (equals.sum()/equals.shape[0]).item()
            
            if iter_id % print_every == 0:
                print(f"\t{iter_id}/{len(testloader)}\ttest_loss: {acc_losses/iter_id:.4f}\taccuracy: {accuracy/iter_id*100:.4f}%")

    return acc_losses/len(testloader), accuracy/len(testloader)
    
