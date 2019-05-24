import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def loss_func(output,label):
    
    batch_size = output.size(0)
#     binary_mask = torch.zeros(batch_size, label.shape[1])

#     for i in range(batch_size):
#         binary_mask[i,:actual_frame[i]] = 1.
#     actual_frame=actual_frame.type(torch.FloatTensor)
    loss=nn.BCELoss(reduction='elementwise_mean')
    m = nn.Sigmoid()
    output_=m(output)
    bce_loss=loss(output_,label)
    # multiple the binary mask with bce loss to mask out the last few zero-padded frames
#     masked_bce_loss = binary_mask * bce_loss  # B, T
    
    # calculate the mean of it for back-propagation, only consider the valid frames
#     average_masked_bce = torch.sum(masked_bce_loss, 1) / actual_frame  # B
#     average_masked_bce = torch.mean(average_masked_bce)
    
    return bce_loss
def mse_loss(output,label)
    loss=nn.MSELoss(reduction='elementwise_mean')
    return loss(output,label)
    
