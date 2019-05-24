import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from loss_func import loss_func

def train(model, epoch,train_loader,cuda, optimizer,log_step,versatile=True):
    start_time = time.time()
    model = model.train()  # set the model to training mode
    train_loss = 0.
    
    # load batch data
    for batch_idx, data in enumerate(train_loader):
        batch_spec = data[0]
        batch_label = data[1]
        batch_length = data[2]
        
        if cuda:
            batch_spec = batch_spec.cuda()
        
        # clean up the gradients in the optimizer
        # this should be called for each batch
        optimizer.zero_grad()
        
        spec_output = model(batch_spec)
        
        # MSE as objective
        loss = loss_func(spec_output,batch_label)
        
        # automatically calculate the backward pass
        loss.backward()
        # perform the actual back-propagation
        optimizer.step()
        
        train_loss += loss.data.item()
        
        # OPTIONAL: you can print the training progress 
        if versatile:
            if (batch_idx+1) % log_step == 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | MSE {:5.4f} |'.format(
                    epoch, batch_idx+1, len(train_loader),
                    elapsed * 1000 / (batch_idx+1), 
                    train_loss / (batch_idx+1)
                    ))
    
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | MSE {:5.4f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    
    return train_loss
        
def validate(model, epoch,validation_loader,cuda):
    start_time = time.time()
    model = model.eval()  # set the model to evaluation mode, this is important if you have BatchNorm in your model!
    validation_loss = 0.
    
    # load batch data
    for batch_idx, data in enumerate(validation_loader):
        batch_spec = data[0]
        batch_label = data[1]
        batch_length = data[2]
        if cuda:
            batch_spec = batch_spec.cuda()
        
        # you don't need to calculate the backward pass and the gradients during validation
        # so you can call torch.no_grad() to only calculate the forward pass, save time and memory
        with torch.no_grad():
        
            spec_output = model(batch_spec)
        
            # MSE as objective
            loss = loss_func(spec_output,batch_label)
        
            validation_loss += loss.data.item()
    
    validation_loss /= (batch_idx+1)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | MSE {:5.4f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)
    
    return validation_loss

