"""
Preprocessing images of MNIST dataset to get a dataset similar to the one used in  Yann LeCun's paper

Notes:
- During the unavailability of the dataset used in the paper, its closest alternate avalibale today, the MNIST dataset, is used.
- A random seed is set is to ensure reproducibility of the results.
- Training dataset consists of 7921 images and corresponding targets.
- Testing dataset consists of 2007 images and corresponding targets.
- Each image in the scaled to 16x16 dimension grayscale using bilinear interpolation method.
- The values of each pixel are scaled to range [-1, 1] from [0,255]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets

np.random.seed(20)
torch.manual_seed(20)

#If i is 0 then we download the training dataset in the loop, if its 2 we download the test dataset
for i in range(2):

  if i==0 :
    data = datasets.MNIST(root ='./data', train =True,  download = True)
    num_samples = 7921
  else:
    data = datasets.MNIST(root ='./data', train =False, download = True)
    num_samples = 2007

  #In order to select the appropriate number of random samples from the dataset, we generate a list of random indices
  rand = np.random.permutation(len(data))
  rand = rand[0:num_samples]

  #Initializing the tensors to store the preprocessed dataset
  X = torch.full((num_samples,1,16,16), -1.0, dtype = torch.float32)
  y = torch.full((num_samples,10), -1.0, dtype = torch.float32)

  for inx, num in enumerate(rand):
    temp = data.data[num]/127.5 -1   #Scaling the values to [-1,1]
    X[inx] = F.interpolate(temp[None, None, :], size = (16,16), mode = 'bilinear')   #Interpolation operation used to resize the images
    y[inx, data.targets[num].to(torch.int)] = 1.0   #The targets are made into tensors of size 10 where the value at the index pertaining to the target is alone set to 1

  torch.save((X,y), f'set_{i}.pt')