"""
Building and training the neural network as specificed in Yann LeCun's 1989 research paper

Notes:
- The convnet has 3 hidden layers and 1 output layer
- H1 has 12 filters of size 5x5 with stride 2 and constant paddinf of -1.
- H2 has 12 filters of size 5x5 and stride 2 but each filter connects only 8 of the 12 input planes.
- H3 is a fully connected layer with 30 neurons
- Output layer is a fully connected layer with 10 neurons.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(20)
torch.manual_seed(20)
torch.use_deterministic_algorithms(True)


class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    MACs = 0
    ACTS = 0

    #Function to initialize weights
    def init_weights(fan_in, *weight_shape):
      rand_nos = (torch.rand(*weight_shape) - 0.5) * (1 - (-1))
      rand_weights = rand_nos * (2.4 / (np.sqrt(fan_in)))
      return rand_weights

    #H1 layer parameter initialization
    self.H1_weights = nn.Parameter(init_weights(5*5*1, 12, 1, 5, 5))
    self.H1_bias = nn.Parameter(torch.zeros(12, 8, 8))

    MACs += (5*5) * (8*8) * 12  #Calculating the number of multiply acculumulate operations
    ACTS += (8*8) * 12  #Calculating the number of activations

    #H2 layer parameter initialization
    self.H2_weights = nn.Parameter(init_weights(5*5*8, 12, 8, 5, 5))
    self.H2_bias = nn.Parameter(torch.zeros(12, 4, 4))

    MACs += (5*5*8) * (4*4) * 12
    ACTS += (4*4) * 12

    #H3 layer parameter initialization. Its a fully connected layer
    self.H3_weights = nn.Parameter(init_weights(4*4*12, 4*4*12, 30))
    self.H3_bias = nn.Parameter(torch.zeros(30))

    MACs += (4*4*12) * (30)
    ACTS += (30)

    #Output layer parameter initialization. Its a fully connected layer
    self.out_weights = nn.Parameter(init_weights(30, 30, 10))
    self.out_bias = nn.Parameter(-torch.ones(10))

    MACs += (30) * (10)
    ACTS += (10)

    self.MACs = MACs
    self.ACTS = ACTS

  def forward(self, img):

    #Shape of the image should be (1, 1, 16, 16)
    img = F.pad(img, (2,2,2,2), 'constant', -1.0)

    #Implementing the layer H1
    H1_temp = F.conv2d(img, self.H1_weights, stride=2) + self.H1_bias
    H1_res = torch.tanh(H1_temp)

    #Implementing the layer H2
    """
    Each neuron in H2 layer connects to 5x5 regions in 8 of the 12 different input planes.
    The scheme for selecting the planes is not specified in the paper.
    We implement 3 seperate convoultions and concatinate the result.
    The first set of neurons connect to first 8 input planes.
    The second set of neurons connect to next 8 input planes.
    The third set of neurons connect to first 4 and last 4 input planes.
    """
    H1_res = F.pad(H1_res, (2,2,2,2), 'constant', -1.0)
    H2_slice1 = F.conv2d(H1_res[:, 0:8], self.H2_weights[0:4], stride=2)   
    H2_slice2 = F.conv2d(H1_res[:, 4:12], self.H2_weights[4:8], stride=2)
    H2_slice3 = F.conv2d(torch.cat((H1_res[:, 0:4], H1_res[:, 8:12]), dim=1), self.H2_weights[8:12], stride=2)
    H2_temp = torch.cat((H2_slice1, H2_slice2, H2_slice3), dim=1) + self.H2_bias   
    H2_res = torch.tanh(H2_temp)

    #Implementing the layer H3
    H2_res = H2_res.flatten(start_dim=1)
    H3_temp = torch.matmul(H2_res, self.H3_weights) + self.H3_bias
    H3_res = torch.tanh(H3_temp)

    #Implementing the output layer
    H_out = torch.matmul(H3_res, self.out_weights) + self.out_bias
    H_out = torch.tanh(H_out)

    return H_out

  def get_MACs(self):
    return self.MACs

  def get_ACTS(self):
    return self.ACTS

if __name__ == '__main__':

  #Creating an instance of the CNN model
  model = CNN()
  print("Model Statistics:")
  print("Parameters: ", sum(p.numel() for p in model.parameters()))
  print("Number of MACs: ", model.get_MACs())
  print("Number of activations : ",model.get_ACTS())

  #Loading the dataset
  x_train, y_train = torch.load('set_0.pt')
  x_test, y_test = torch.load('set_1.pt')

  #We use SGD as the exact loss function optimization technique is not mentioned in the paper
  optimizer = optim.SGD(model.parameters(), lr =0.03)

  #Evaluating the training and testing process
  def eval_split(split):
    model.eval()
    img_eval, label_eval = (x_train, y_train) if split == 'train' else (x_test, y_test)
    y_pred_eval = model(img_eval)
    loss_eval = torch.mean((y_pred_eval - label_eval)**2)
    err = torch.mean((label_eval.argmax(dim=1) != y_pred_eval.argmax(dim=1)).float())
    print(f"eval: split {split:5s}. loss {loss_eval.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*label_eval.size(0))}")

  #Training implementation
  for pass_num in range(23):

    model.train()

    for step_num in range(x_train.size(0)):

        img, label = x_train[[step_num]], y_train[[step_num]]
        y_pred = model(img)
        loss = torch.mean((label-y_pred)**2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(pass_num + 1)
    eval_split('train')
    eval_split('test')