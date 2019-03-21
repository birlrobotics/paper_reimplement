from torch.autograd import Variable
import torch
import numpy as np

x_data = np.array((1.0,2.0,3.0))
y_data = np.array(((2.0,4.0,6.0),(3.0,6.0,9.0)))

x_data = torch.tensor(x_data).reshape(1,3).float()
y_data = torch.tensor(y_data).reshape(2,3).float()

print(y_data)
w = torch.randn(2).reshape(2,1)
w.requires_grad = True
print(w)

def forward(x):
    return torch.matmul(w , x)

forward(x_data)

def loss(x,y):
    y_pred = forward(x)
    y_loss = 0
    for y_p, y_r in zip(y_pred.view(-1),y.view(-1)):
        y_loss = (y_p-y_r)**2
    return y_loss

l = loss(x_data,y_data)

print(l)