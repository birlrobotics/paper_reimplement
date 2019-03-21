from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


x_data = np.array((1.0,2.0,3.0,4.0,5.0))
y_data = np.array(((2.0,3.0),(4.0,6.0),(6.0,9.0),(8.0,12.0),(10.0,15.0)))

x_data = torch.tensor(x_data).reshape(5,1).float()
y_data = torch.tensor(y_data).reshape(5,2).float()

w = torch.randn(2).reshape(1,2)

w.requires_grad = True

print(w.size())
print(w)

def forward(x):
    return torch.matmul(x , w)

def loss(x,y):
    y_pred = forward(x)
    y_loss = 0
    for y_p, y_r in zip(y_pred.view(-1),y.view(-1)):
        y_loss += (y_p-y_r)**2
    return y_loss

input_data = torch.tensor((5.0,6.0,7.0)).reshape(3,1)
print(input_data)
print("predict (before training)", 4 , forward(input_data))

epoch = range(1,30)
for i in epoch:
    for x, y in zip(x_data,y_data):
        l = F.mse_loss(forward(x),y)
        l.backward()
        
        
        w.data = w.data - 0.01* w.grad.data
        print("w.data:",w.data,"w.grad:", w.grad)
        w.grad.data.zero_()
        
    print("epoch:",i, ", loss:",l)
    
print("predict (after training)", 4 , forward(input_data).data)