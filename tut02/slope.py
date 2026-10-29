import torch
import torch.utils.data
import numpy as np

lr = 1.9 # learning rate
mom = 0.0 # momentum

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = torch.nn.Parameter(torch.zeros((1),
                                    requires_grad=True))
    def forward(self, input):
        output  = self.w * input
        return(output)

device = 'gpu'

input  = torch.Tensor([[1]])
target = torch.Tensor([[1]])

slope_dataset = torch.utils.data.TensorDataset(input,target)
train_loader  = torch.utils.data.DataLoader(slope_dataset,batch_size=1)

# create neural network according to model specification
net = MyModel().to(device) # CPU or GPU

# choose between SGD, Adam or other optimizer
optimizer=torch.optim.SGD(net.parameters(),lr=lr,momentum=mom)

epochs = 1000

epoch = 0
while epoch < epochs:
    epoch = epoch + 1
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad() # zero the gradients
        output = net(data)    # apply network
        loss = 0.5*torch.mean((output-target)*(output-target))
        print('Ep%3d: zero_grad(): w.data=%7.4f loss=%7.4f' \
                     % (epoch, net.w.data, loss))
        loss.backward()       # compute gradients
        optimizer.step()      # update weights
        print('            step(): w.grad=%7.4f w.data=%7.4f' \
                     % (net.w.grad, net.w.data))
        if loss < 0.000000001 or np.isnan(loss.data.item()):
            epoch = epochs
            break