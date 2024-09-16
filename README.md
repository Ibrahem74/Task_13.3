# Neural Networks Overview

## 1. Neural Networks

Neural networks simulate the human brain and learn from examples.

## 2. Perceptrons
A perceptron:
- Uses weight values to make decisions.
- Takes different inputs and can have many layers of inputs to provide one output.
- The output is determined by the following function:


- The bias is used instead of a threshold and equals `-threshold`. It measures how easy it is to get the perceptron to output 1.
- Perceptrons can be used to implement logic functions such as the NAND gate, which is a universal gate.

## 3. Sigmoid Neurons

Sigmoid neurons are similar to perceptrons but can output any value between 0 and 1. This helps to overcome the problem where a small change in input can cause a big change in output. The output function is:
- This can be represented using a step function, which makes it similar to perceptrons.
- The change in output is approximately:

## 4. Architecture of Neural Networks

Neural network architecture includes:
- **Input Layer**
- **Hidden Layers** (not input or output layers)
- **Output Layer**

We use feedforward neural networks, where the output from one layer is used as input to the next layer, and no loops are used.

## 5. Classify Handwritten Digits

To classify handwritten digits, we separate each digit and recognize individual digits using a three-layer neural network:
- **Input Layer**: 28 x 28 neurons
- **Hidden Layer**: 15 neurons
- **Output Layer**: 10 neurons

## 6. Learning with Gradient Descent

Gradient descent is used to train neural networks by minimizing a cost function, such as the quadratic cost (mean squared error). It helps in finding the optimal weights and biases to minimize the cost.

### Process Overview

- **Cost Function**: For neural networks, the quadratic cost function \( C(w, b) \) quantifies the difference between the network’s predicted output and the true output for a given input. The aim is to minimize this difference.

- **Why Gradient Descent**: Directly minimizing the number of misclassified images is not feasible due to the non-smooth nature of that function. Instead, we minimize the smooth quadratic cost, making it easier to adjust the network’s weights and biases.

- **Gradient Descent**:
- **Initialization**: Start with random weights and biases.
- **Compute Gradient**: Calculate the gradient of the cost function with respect to the weights and biases.
- **Update Weights and Biases**: Adjust the weights and biases in the direction that reduces the cost:

  ```
  v' = v - η ∇C
  ```

  where \( η \) is the learning rate (controls step size) and \( ∇C \) is the gradient of the cost function.

- **Iterate**: Repeat the process until the cost function converges to a minimum.

- **Learning Rate \( η \)**: Choosing the right learning rate is critical. If \( η \) is too large, the algorithm may overshoot the minimum; if it’s too small, convergence will be slow.

By applying this method repeatedly, gradient descent allows neural networks to "learn" by optimizing their internal parameters (weights and biases) to minimize prediction error.
