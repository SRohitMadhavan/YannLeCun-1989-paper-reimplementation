
# Reimplementation of LeCun's 1989 paper

-The code in the repository aims to reimplement the iconic [Backpropagation Applied to Handwritten Zip COde Recognition] (http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) research paper published by Yann LeCun et al. in 1989.

#### Dataset

-Due to the unavailability of the exact dataset used in the paper, we take the MNIST dataset which, to my knowledge, is the closest dataset to the original dataset used in the paper.

-We randomly select the 7291 training and 2007 test samples from the MNIST dataset and then resize them to 16x16 to get a good approximation of the dataset mentioned in the paper.

```
$ python preprocessing.py
```

#### Implementation of the NN and training

![teaser](architecture.png)

-The neural network is implemented as specified in the paper with few assumpitions being made at parts where information was not specified in the paper. 

```
$ python implementation.py
```

#### Notes from the paper

- Training dataset consists of 7921 digits
- Testing dataset consists of 200y digits
- Each image is preprocessed to a 16x16 image in grayscale. The range of each pixel is [-1, 1]
- Neural network has3 hidden layers H1, H2, H3 and 1 output layer
- H1 layer has 12 5x5 filters with stride 2. Constant padding of -1
- Units do not share biases.(including in the same feature plane)
- H2 layer has 12 5x5 filters with stride 2, each connecting to 8 out of the 12 different input planes.
- H3 layer is a fully connected layer consisting 30 neurons.
- Output layer is a fully connected layer consisting 10 neurons.
- Tanh activation function is applied on all layers (including output layers).
- The cost function is mean squared error.
- The initialized weights are in the range of U[-2.4/F, 2.4/F] where F is the fan-in(number of inputs into a neuron).
- Patters presented in constant order during training.
- Use a special version of Newton's algorithm that uses a positive, diagonal approzimation of Hessian
- Trained for 23 epochs.






