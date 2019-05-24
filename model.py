import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # layers, activation function, and normalization
        self.freq_dim=24
        self.hidden_unit =[256,128,64]  # number of hidden units
        
        # input layer
        self.layer1 = nn.Linear(self.freq_dim, self.hidden_unit[0])
        self.activation1 = nn.ReLU()
        self.norm = nn.BatchNorm1d(self.hidden_unit[0])
        
        # you can combine them
        self.layer1 = nn.Sequential(self.layer1,
                                    self.activation1,self.norm)
        
        # do the same for other two layers
        # hidden layer
        self.layer2 = nn.Sequential(nn.Linear(self.hidden_unit[0], self.hidden_unit[1]),
                                    nn.ReLU(),nn.BatchNorm1d(self.hidden_unit[1])
                                   )
        # output layer
        # note that since the input is unbounded, the output layer should be a linear layer
        self.layer3 = nn.Sequential(nn.Linear(self.hidden_unit[1], 5),nn.Softmax(dim=1))
        
    # the function for the forward pass of network (i.e. from input to output)
    def forward(self,input):
        # the input is 3-dimensional, (batch, freq, time)
        # we need to reshape it into 2-dimensional, (batch*time, freq)
        
        batch_size = input.size(0)
        freq = input.size(1)
        num_frame = input.size(2)
        
        input = input.transpose(1, 2).contiguous()  # (batch, time, freq)
        input = input.view(batch_size*num_frame, freq)  # (batch*time, freq)
        
        # pass it through layers
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
       
        
        # reshape back
        output = output.view(batch_size, num_frame,5)
        
        
        return output
    
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.hidden_unit = 256 # number of hidden units
        self.freq_dim=24
        # input layer
        self.lstm_layer = nn.LSTM(input_size=self.freq_dim,
                                  hidden_size=self.hidden_unit,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)
        
        self.output_linear = nn.Linear(self.hidden_unit, 5)
        self.fc_layer= nn.Sequential(self.output_linear, nn.Softmax(dim=1))
    # the function for the forward pass of network (i.e. from input to output)
    def forward(self, input):
        # note that LSTM in Pytorch requires input shape as (batch, seq, feature)
        # so we need to reshape the input
        
        batch_size = input.size(0)
        freq = input.size(1)
        num_frame = input.size(2)
        
        input = input.transpose(1, 2).contiguous()  # (batch, time, freq)
        
        # pass it through layers
        output, (h_n, c_n) = self.lstm_layer(input)  # (batch, time, freq)
        output = output.contiguous().view(batch_size*num_frame, -1)  # (batch*time, freq)
        output = self.fc_layer(output)
        
        # reshape back
        output = output.view(batch_size, num_frame, 5)
        
        
        return output
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        #self.hidden_unit = [256,128] # number of hidden units
        #self.freq_dim=24
        # input layer
        
        self.cov1=nn.Conv2d(1,16,7)
        self.cov2=nn.Conv2d(16,32,5)
        self.cov3=nn.Conv2d(32,64,3)
        self.cov4=nn.Conv2d(64,128,3)
        self.cov5=nn.Conv2d(128,256,3)
        self.layer1=nn.Sequential(self.cov1,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(16))
        self.layer2=nn.Sequential(self.cov2,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(32))
        self.layer3=nn.Sequential(self.cov3,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(64))
        self.layer4=nn.Sequential(self.cov4,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(128))
        self.layer5=nn.Sequential(self.cov5,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(256))
        #self.layer5=nn.Sequential(self.cov5,nn.ReLU(),
         #                          nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(256))
        self.fc1=nn.Sequential(nn.Linear(29*256,128),nn.ReLU(),nn.BatchNorm1d(128), nn.Dropout(p=0.5))
        #self.fc2=nn.Linear(128,64)
        self.fc2=nn.Sequential(nn.Linear(128,5),nn.Softmax(dim=1))
        
    # the function for the forward pass of network (i.e. from input to output)
    def forward(self, input):
        # note that LSTM in Pytorch requires input shape as (batch, seq, feature)
        # so we need to reshape the input
        
        batch_size = input.size(0)
        freq = input.size(1)
        num_frame = input.size(2)
        input=input.unsqueeze(1).contiguous()
        #input = input.transpose(1, 2).contiguous()  # (batch, time, freq)
        
        # pass it through layers
        output=self.layer1(input)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=self.layer5(output)
        output = output.contiguous().view(batch_size,-1)  # (batch*time, freq)
        output = self.fc1(output)
        output=self.fc2(output)
        #output=self.fc3(output)
        # reshape back
        output = output.view(batch_size,5)
        return output
class CNN_rnn(nn.Module):
    def __init__(self):
        super(CNN_rnn, self).__init__()
        
        #self.hidden_unit = [256,128] # number of hidden units
        #self.freq_dim=24
        # input layer
        #self.blstm=nn.LSTM(input_size=, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        
        
        self.cov1=nn.Conv2d(1,16,7)
        self.cov2=nn.Conv2d(16,32,5)
        self.cov3=nn.Conv2d(32,64,3)
        self.cov4=nn.Conv2d(64,128,3)
        self.cov5=nn.Conv2d(128,256,3)
        self.layer1=nn.Sequential(self.cov1,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(16))
        self.layer2=nn.Sequential(self.cov2,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(32))
        self.layer3=nn.Sequential(self.cov3,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(64))
        self.layer4=nn.Sequential(self.cov4,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(128))
        self.layer5=nn.Sequential(self.cov5,nn.ReLU(),
                                   nn.MaxPool2d(2,stride=2),nn.BatchNorm2d(256))
        self.bilstm=nn.LSTM( input_size=256,hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1=nn.Sequential(nn.Linear(256*2,256),nn.ReLU(),nn.BatchNorm1d(256), nn.Dropout(p=0.5))
        
        self.fc2=nn.Sequential(nn.Linear(256,5),nn.Softmax(dim=1))
        #self.fc_layer= nn.Sequential(self.output_linear, nn.Softmax(dim=1))
    # the function for the forward pass of network (i.e. from input to output)
    def forward(self, input):
        # note that LSTM in Pytorch requires input shape as (batch, seq, feature)
        # so we need to reshape the input
        
        batch_size = input.size(0)
        freq = input.size(1)
        num_frame = input.size(2)
        
        #input = input.transpose(1, 2).contiguous()  # (batch, time, freq)
        input=input.unsqueeze(1).contiguous()
        # pass it through layers
        output=self.layer1(input)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=self.layer5(output)
        output=output.permute(0,3,2,1).contiguous() # (batch,W, H,channel)
        N,C,H,W=output.shape
        
        output = output.view(N,C,W*H).contiguous()
        
        output, (h_n, c_n) = self.bilstm(output) # (batch,W,H*channel)to (batch,W,256)
       
        output=output[:,-1,:].contiguous() #(batch,1,H*channel)
       
        output = self.fc1(output)
        output=self.fc2(output)
        
        output = output.view(-1,5)
        return output
        
