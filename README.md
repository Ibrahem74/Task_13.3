# NN Magic (Individual)

## About

Neural Networks are a class of machine learning algorithms inspired by the way the human brain functions. They can process complex data and learn from experience, making them highly effective in tasks such as image recognition, speech processing, and autonomous systems. In this task, you will delve into the fundamentals of neural networks by building one from scratch, exploring key concepts, and applying them to a real-world problem: classifying handwritten digits from the MNIST dataset.

## Requirements

### 1. Neural Network from Scratch

- **Objective:** Build a neural network based on foundational concepts introduced in Michael Nielsen’s book *Neural Networks and Deep Learning* (Chapter 1). You will implement the network from scratch without using any deep learning libraries.
- **Concepts to Explore:**
  - **Perceptrons:** Simulate basic decision-making units that take inputs, apply weights, and use an activation function to produce an output.
  - **Bias and Threshold:** Understand how bias simplifies the perceptron’s decision process, replacing the threshold with a bias term.
  - **Sigmoid Neurons:** Learn about sigmoid activation functions that output values between 0 and 1, addressing issues of abrupt output changes in perceptrons.
  - **Neural Network Architecture:** Build a network with input layers, hidden layers, and output layers. Focus on feedforward networks where each layer’s output serves as input to the next, avoiding loops.
  - **Gradient Descent:** Implement gradient descent to minimize the cost function (such as mean squared error) and optimize weights and biases.
- **Deliverable:** Summarize your learnings in a README file as you work through the chapter.

### 2. MNIST Classification

- **Objective:** Use the neural network you built to classify handwritten digits from the MNIST dataset.
- **Requirements:** Ensure your model can accurately recognize and predict digit labels (0-9) by training on the MNIST images.

### 3. Build with PyTorch or TensorFlow

- **Objective:** After implementing the neural network from scratch, create an equivalent model using either PyTorch or TensorFlow.
- **Tasks:**
  - Compare the two implementations in terms of ease of use, performance, and flexibility.
  - Document your comparisons and findings in the README file.

## Key Concepts and Formulas

- **Perceptrons:**  
  The perceptron makes a decision based on weighted inputs and a bias term. The output is determined as follows:
  ```plaintext
  output = { 0 if ∑j wj xj ≤ bias
           1 if ∑j wj xj > bias }
