# digitRecogniser
neural network to detect digits

I have written the same logic with 2 implementations
one from scratch with numpy
other using pytorch

Link to Kaggle notebook
https://www.kaggle.com/sairushikjasti/digitrecogniser
https://www.kaggle.com/code/sairushikjasti/digitrecog-pytorch

input layer: 784 nodes
1st layer: 10 nodes
output layer: 10 nodes

## overview of forward propagation ##
let A0 be the input image
A1 = relu(A0.W1 + b1)
A2 = norm(A1.W2 + b2)
A2 is the output probability of each digit for input image
