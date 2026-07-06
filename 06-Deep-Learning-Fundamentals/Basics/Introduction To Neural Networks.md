---
title: "Introduction To Neural Networks"
source: "https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/"
author:
  - "[[GeeksforGeeks]]"
published: 2019-01-17
created: 2026-07-06
description: "Your All-in-One Learning Portal: GeeksforGeeks is a comprehensive educational platform that empowers learners across domains-spanning computer science and programming, school education, upskilling, commerce, software tools, competitive exams, and more."
tags:
  - "clippings"
---
Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns directly from data without relying on pre-defined rules, enabling them to perform tasks such as pattern recognition and decision-making.

![bhu.webp](https://media.geeksforgeeks.org/wp-content/uploads/20250703153812268928/bhu.webp) ![bhu.webp](https://media.geeksforgeeks.org/wp-content/uploads/20250703153812268928/bhu.webp) ![history_and_evolution_of_neural_networks.webp](https://media.geeksforgeeks.org/wp-content/uploads/20251213153112140680/history_and_evolution_of_neural_networks.webp) ![history_and_evolution_of_neural_networks.webp](https://media.geeksforgeeks.org/wp-content/uploads/20251213153112140680/history_and_evolution_of_neural_networks.webp) ![types_of_neural_networks.webp](https://media.geeksforgeeks.org/wp-content/uploads/20251213153112518597/types_of_neural_networks.webp) ![types_of_neural_networks.webp](https://media.geeksforgeeks.org/wp-content/uploads/20251213153112518597/types_of_neural_networks.webp) ![key_applications_of_deep.webp](https://media.geeksforgeeks.org/wp-content/uploads/20251213153112304250/key_applications_of_deep.webp) ![key_applications_of_deep.webp](https://media.geeksforgeeks.org/wp-content/uploads/20251213153112304250/key_applications_of_deep.webp)

3 / 4

****Neural Networks are built from several key components:****

- ****Neurons****: The basic units that receive inputs, each neuron is governed by a threshold and an activation function.
- ****Connections****: Links between neurons that carry information, regulated by weights and biases.
- ****Weights and Biases****: These parameters determine the strength and influence of connections.
- ****Propagation Functions****: Mechanisms that help process and transfer data across layers of neurons.
- ****Learning Rule****: The method that adjusts weights and biases over time to improve accuracy.

****Learning in neural networks follows a structured, three-stage process:****

1. ****Input Computation****: Data is fed into the network.
2. ****Output Generation****: Based on the current parameters, the network generates an output.
3. ****Iterative Refinement****: The network refines its output by adjusting weights and biases, gradually improving its performance on diverse tasks.
![420047197](https://media.geeksforgeeks.org/wp-content/uploads/20260414151507594095/420047197.webp)

The image illustrates the analogy between a biological neuron and an artificial neuron, showing how inputs are received and processed to produce outputs in both systems.

## Importance of Neural Networks

- ****Identify Complex Patterns:**** Recognize intricate structures and relationships in data; adapt to dynamic and changing environments.
- ****Learn from Data:**** Handle vast datasets efficiently; improve performance with experience and retraining.
- ****Drive Key Technologies:**** Power natural language processing (NLP); enable self-driving vehicles; support automated decision-making systems.
- ****Boost Efficiency:**** Streamline workflows and processes; enhance productivity across industries.
- ****Backbone of AI:**** Serve as the core driver of artificial intelligence progress; continue shaping the future of technology and innovation.

## Layers in Neural Network Architecture

![_neural_network](https://media.geeksforgeeks.org/wp-content/uploads/20251213151009129404/_neural_network.webp)

Layers in Neural Network Architecture

1. ****Input Layer:**** This is where the network receives its input data. Each input neuron in the layer corresponds to a feature in the input data.
2. ****Hidden Layers:**** These layers perform most of the computational heavy lifting. A neural network can have one or multiple hidden layers. Each layer consists of units (neurons) that transform the inputs into something that the output layer can use.
3. ****Output Layer:**** The final layer produces the output of the model. The format of these outputs varies depending on the specific task like classification, regression.

## Working of Neural Networks

### 1\. Forward Propagation

When data is input into the network, it passes through the network in the forward direction, from the input layer through the hidden layers to the output layer. This process is known as forward propagation. Here’s what happens during this phase:

****1\. Linear Transformation:**** Each neuron in a layer receives inputs which are multiplied by the weights associated with the connections. These products are summed together and a bias is added to the sum. This can be represented mathematically as:

> $$
> z = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b
> $$

where

- $$
	w
	$$
	represents the weights
- $$
	x
	$$
	represents the inputs
- $$
	b
	$$
	is the bias

****2\. Activation:**** The result of the linear transformation (denoted as

$$
z
$$

) is then passed through an activation function. The activation function is crucial because it introduces non-linearity into the system, enabling the network to learn more complex patterns. Popular activation functions include ReLU, sigmoid and tanh.

### 2\. Backpropagation

After forward propagation, the network evaluates its performance using a loss function which measures the difference between the actual output and the predicted output. The goal of training is to minimize this loss. This is where backpropagation comes into play:

- ****Loss Calculation:**** The network calculates the loss which provides a measure of error in the predictions. The loss function could vary; common choices are mean squared error for regression tasks or cross-entropy loss for classification.
- ****Gradient Calculation:**** The network computes the gradients of the loss function with respect to each weight and bias in the network. This involves applying the chain rule of calculus to find out how much each part of the output error can be attributed to each weight and bias.
- ****Weight Update:**** Once the gradients are calculated, the weights and biases are updated using an optimization algorithm like stochastic gradient descent (SGD). The weights are adjusted in the opposite direction of the gradient to minimize the loss. The size of the step taken in each update is determined by the learning rate.

### 3\. Iteration

This process of forward propagation, loss calculation, backpropagation and weight update is repeated for many iterations over the dataset. Over time, this iterative process reduces the loss and the network's predictions become more accurate.

> Through these steps, neural networks can adapt their parameters to better approximate the relationships in the data, thereby improving their performance on tasks such as classification, regression or any other predictive modeling.

## Example of Email Classification

Let's consider a record of an email dataset:

| Email ID | Email Content | Sender | Subject Line | Label |
| --- | --- | --- | --- | --- |
| 1 | "Get free gift cards now!" | spam@example.com | "Exclusive Offer" | 1 |

To classify this email, we will create a feature vector based on the analysis of keywords such as "free" "win" and "offer"

****The resulting feature vector is:****

- ****free:**** Present (1)
- ****win:**** Absent (0)
- ****offer:**** Present (1)

****Feature Vector:****

> \[1, 0, 1\]

The feature vector is passed through the neural network, where each neuron computes a weighted sum of the inputs and applies an activation function to produce its output.

****Hidden Layer Calculation:****

Assume the hidden layer contains two neurons with the following weights

- Neuron H1: \[ 0.5, −0.2, 0.3\]
- Neuron H2: \[ 0.4, 0.1, −0.5\]

For the input vector \[1, 0, 1\], the weighted sums are:

- ****For H1: (****1 × 0.5) +(0 ×− 0.2) + ( 1 × 0.3) = 0.5 + 0+ 0.3 = 0.8
- ****For H2:**** (1 × 0.4) +(0 × 0.1) + ( 1 × −0.5) = 0.4 + 0 −0.5 =−0.1

****Applying the**** [****ReLU****](https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/) ****activation function:****

- ****H1 Output:**** ReLU(0.8) = 0.8
- ****H2 Output:**** ReLu(-0.1) = 0

****Output Layer:**** The outputs from the hidden layer are then passed to the output neuron.

- ****Output Weights:**** \[0.7, 0.2\]
- ****Input from Hidden Layer:**** \[0.8, 0\]
- ****Weighted Sum:**** (0.8×0.7)+(0×0.2)=0.56+0=0.56
- ****Activation (Sigmoid):****
	$$
	\sigma(0.56) = \frac{1}{1 + e^{-0.56}} \approx 0.636
	$$

****Final Classification:****

- The output value of approximately ****0.636**** indicates the probability of the email being spam.
- Since this value is greater than 0.5, the neural network classifies the email as spam (1).
![backpropagation_in_neural_network_8](https://media.geeksforgeeks.org/wp-content/uploads/20251213153337823586/backpropagation_in_neural_network_8.webp)

Neural Network

## Learning of a Neural Network

### 1\. Learning with Supervised Learning

In supervised learning, a neural network learns from labeled input-output pairs provided by a teacher. The network generates outputs based on inputs and by comparing these outputs to the known desired outputs, an error signal is created. The network iteratively adjusts its parameters to minimize errors until it reaches an acceptable performance level.

### 2\. Learning with Unsupervised Learning

Unsupervised learning involves data without labeled output variables. The primary goal is to understand the underlying structure of the input data (X). Unlike supervised learning, there is no instructor to guide the process. Instead, the focus is on modeling data patterns and relationships, with techniques like clustering and association commonly used.

### 3\. Learning with Reinforcement Learning

Reinforcement learning enables a neural network to learn through interaction with its environment. The network receives feedback in the form of rewards or penalties, guiding it to find an optimal policy or strategy that maximizes cumulative rewards over time. This approach is widely used in applications like gaming and decision-making.

## Types of Neural Networks

There are several types of neural networks, including:

- [****Feedforward Networks****](https://www.geeksforgeeks.org/deep-learning/feedforward-neural-network/)****:**** It is a simple artificial neural network architecture in which data moves from input to output in a single direction.
- [****Singlelayer Perceptron:****](https://www.geeksforgeeks.org/python/single-layer-perceptron-in-tensorflow/) It has one layer and it applies weights, sums inputs and uses activation to produce output.
- [****Multilayer Perceptron (MLP)****](https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/)****:**** It is a type of feedforward neural network with three or more layers, including an input layer, one or more hidden layers and an output layer.
- [****Convolutional Neural Network (CNN)****](https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/)****:**** It is designed for image processing. It uses convolutional layers to automatically learn features from input images, enabling effective image recognition and classification.
- [****Recurrent Neural Network (RNN)****](https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/)****:**** Handles sequential data using feedback loops to retain context over time.
- [****Long Short-Term Memory (LSTM)****](https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/)****:**** A type of RNN with memory cells and gates to handle long-term dependencies and avoid vanishing gradients.

## Implementation of Neural Network using TensorFlow

Here, we implement simple feedforward neural network that trains on a sample dataset and makes predictions using following steps:

### Step 1: Import Necessary Libraries

Import necessary libraries, primarily [TensorFlow](https://www.geeksforgeeks.org/python/introduction-to-tensorflow/) and [Keras](https://www.geeksforgeeks.org/deep-learning/what-is-keras/), along with other required packages such as [NumPy](https://www.geeksforgeeks.org/python/introduction-to-numpy/) and [Pandas](https://www.geeksforgeeks.org/pandas/introduction-to-pandas-in-python/) for data handling.

```elm
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### Step 2: Create and Load Dataset

- Create or load a dataset. Convert the data into a format suitable for training (usually NumPy arrays).
- Define features (X) and labels (y).

```prolog
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values
```

### Step 3: Create a Neural Network

Instantiate a Sequential model and add layers. The input layer and hidden layers are typically created using `Dense` layers, specifying the number of neurons and activation functions.

```
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### Step 4: Compiling the Model

Compile the model by specifying the loss function, optimizer and metrics to evaluate during training. Here we will use [binary crossentropy](https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/) and [adam optimizer](https://www.geeksforgeeks.org/deep-learning/adam-optimizer/).

```
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
```

### Step 5: Train the Model

Fit the model on the training data, specifying the number of epochs and batch size. This step trains the neural network to learn from the input data.

```
model.fit(X, y, epochs=100, batch_size=1, verbose=1)
```

### Step 6: Make Predictions

Use the trained model to make predictions on new data. Process the output to interpret the predictions like converting probabilities to binary outcomes.

```
test_data = np.array([[0.2, 0.4]])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)
```

****Output:****

> Predicted label: 1

## Applications

1. ****Image and Video Recognition****: CNNs are extensively used in applications such as facial recognition, autonomous driving and medical image analysis.
2. ****Natural Language Processing (NLP)****: RNNs and transformers power language translation, chatbots and sentiment analysis.
3. ****Finance****: Predicting stock prices, fraud detection and risk management.
4. ****Healthcare****: Neural networks assist in diagnosing diseases, analyzing medical images and personalizing treatment plans.
5. ****Gaming and Autonomous Systems****: Neural networks enable real-time decision-making, enhancing user experience in video games and enabling autonomous systems like self-driving cars.

## Advantages

- ****Can model complex relationships:**** Neural networks use non-linear activation functions to learn intricate patterns that cannot be captured by linear models.
- ****Automatic feature learning:**** They automatically extract relevant features from raw data, reducing the need for manual feature engineering.
- ****Supports parallel processing:**** Multiple neurons can perform computations simultaneously, improving computational efficiency, especially on GPUs.
- ****Generalizes to unseen data:**** Once trained, neural networks can make accurate predictions on new data that was not part of the training set.

## Limitations

- Training large neural networks is computationally intensive and requires significant processing power.
- They act as black box models, making it difficult to understand how decisions are made, which can be a concern in critical applications.
- They may suffer from overfitting, where the model memorizes training data instead of learning general patterns, even though regularization can help reduce this issue.
- Neural networks often require large, well-labeled datasets for effective training and performance may suffer if the data is limited or biased.

