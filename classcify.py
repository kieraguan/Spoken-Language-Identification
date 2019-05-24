import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
def class_accuracy(loader,model,cuda):
    accuracy=0
    number=0
    # the label index for the predicted output probability
    for batch_idx, data in enumerate(loader):
        batch_spec = data[0]
        batch_label = data[1]
        if cuda:
            batch_spec = batch_spec.cuda()
            batch_label=batch_label.cuda()
        with torch.no_grad():
            spec_output = model(batch_spec)
        #predicted_label_index = torch.argmax(spec_output,dim=2)
        #target_label_index = torch.argmax(batch_label,dim=2)
        predicted_label_index = torch.argmax(spec_output,dim=1)
        target_label_index = torch.argmax(batch_label,dim=1)
        correct_frame = torch.sum(predicted_label_index == target_label_index)
        #number+= len(predicted_label_index)*batch_spec.shape[1]
        number+= len(predicted_label_index)
        accuracy += correct_frame.item()

    
    return accuracy/number
def test_accuracy(loader,model,cuda):
    record=np.zeros((6,6))
    for batch_idx, data in enumerate(loader):
        batch_spec = data[0]
        batch_label = data[1]
        if cuda:
            batch_spec = batch_spec.cuda()
            batch_label=batch_label.cuda()
        with torch.no_grad():
            spec_output = model(batch_spec)
        predicted_label_index = torch.argmax(spec_output,dim=1)
        target_label_index = torch.argmax(batch_label,dim=1)
        a=predicted_label_index.numpy()  
        b=target_label_index.numpy()  
        for i in range(a.shape[0]):
            record[a[i],b[i]]+=1
    return record
