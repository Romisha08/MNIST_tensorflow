# MNIST Handwritten Digit Classification

A deep neural network project for classifying handwritten digits from the **MNIST dataset**.

MNIST is often considered the **"Hello World" of Machine Learning** because it is a highly visual, intuitive, and well-preprocessed problem. It is simple enough to understand and implement, while also providing a natural path toward more advanced architectures such as Convolutional Neural Networks (CNNs).

---

##  Project Overview

The objective of this project is to build a deep neural network that takes an image of a handwritten digit as input and correctly determines **which number is shown in the image**.

The MNIST dataset contains:

* **70,000 handwritten digit images**
* **10 classes**, representing digits from `0` to `9`
* Images of size **28 × 28 pixels**
* **Grayscale** images
* Each image contains **784 pixels** (`28 × 28 = 784`)
* Pixel intensity values ranging from **0 to 255**

The problem can therefore be formulated as a **10-class classification problem**.

> **Input:** Image of a handwritten digit
> **Output:** Predicted digit from `0` to `9`

---

##  Why MNIST?

MNIST is a popular starting point for learning Deep Learning because:

* It is a very **visual and intuitive problem**
* The dataset is already **well-preprocessed**
* It is relatively easy to train a neural network on
* It introduces fundamental concepts such as:

  * Neural networks
  * Forward propagation
  * Activation functions
  * Softmax
  * One-hot encoding
  * Loss functions
  * Optimization
  * Backpropagation
  * Validation and testing
* The same concepts can later be extended to more complex problems and architectures such as **CNNs**

---

# 📊 Understanding the Data

Each MNIST image has a resolution of:

```text
28 × 28 pixels
```

Therefore:

```text
28 × 28 = 784 pixels
```

Since MNIST images are grayscale, every pixel represents the **intensity of the color**.

A simplified interpretation is:

```text
0   → Black
255 → White
```

Values between `0` and `255` represent different shades of gray.

For example:

```text
0     → Completely black
128   → Gray
255   → Completely white
```

Each image can therefore be represented as a vector containing **784 numerical values**.

---

#  From Image to Neural Network Input

A neural network expects numerical inputs.

Instead of feeding the network a 2D image:

```text
28 × 28
```

we flatten it into a one-dimensional vector:

```text
784 × 1
```

Conceptually:

```text
28 × 28 image
       ↓
Flatten
       ↓
784-dimensional vector
       ↓
Neural Network
```

Therefore, **each pixel becomes one input feature**.

If we represent the input image as:

```text
X = [x₁, x₂, x₃, ..., x₇₈₄]
```

then:

* `x₁` is the intensity of the first pixel
* `x₂` is the intensity of the second pixel
* ...
* `x₇₈₄` is the intensity of the last pixel

The neural network therefore receives **784 inputs for every image**.

---

#  Neural Network Architecture

The basic deep neural network will consist of:

```text
Input Layer
    ↓
Hidden Layer 1
    ↓
Hidden Layer 2
    ↓
Output Layer
```

### Input Layer

The input layer contains:

```text
784 input units
```

because every MNIST image contains 784 pixels.

Each input corresponds to the intensity of one pixel.

---

## Hidden Layers

We will use **two hidden layers**.

The neurons in these layers perform a linear combination of their inputs followed by a non-linear activation function.

Conceptually:

```text
Weighted Sum
     ↓
Activation Function
     ↓
Neuron Output
```

For a neuron, the basic operation can be represented as:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

where:

* `x` = input values
* `w` = weights
* `b` = bias
* `z` = weighted sum

An activation function is then applied:

```text
a = f(z)
```

The non-linearity allows the neural network to learn complex relationships between pixels and digit classes.

---

#  Output Layer

There are **10 possible digits**:

```text
0 1 2 3 4 5 6 7 8 9
```

Therefore, the output layer contains:

```text
10 output units
```

Each output unit corresponds to one digit.

```text
Output 0 → Digit 0
Output 1 → Digit 1
Output 2 → Digit 2
...
Output 9 → Digit 9
```

---

#  One-Hot Encoding

The target labels will be represented using **one-hot encoding**.

For example, the digit `0` is represented as:

```text
[1 0 0 0 0 0 0 0 0 0]
```

The digit `5` is represented as:

```text
[0 0 0 0 0 1 0 0 0 0]
```

Similarly, the digit `9` would be:

```text
[0 0 0 0 0 0 0 0 0 1]
```

Only the position corresponding to the correct digit contains `1`; all other positions contain `0`.

### One-Hot Encoding Table

| Digit | One-Hot Representation  |
| ----- | ----------------------- |
| 0     | `[1 0 0 0 0 0 0 0 0 0]` |
| 1     | `[0 1 0 0 0 0 0 0 0 0]` |
| 2     | `[0 0 1 0 0 0 0 0 0 0]` |
| 3     | `[0 0 0 1 0 0 0 0 0 0]` |
| 4     | `[0 0 0 0 1 0 0 0 0 0]` |
| 5     | `[0 0 0 0 0 1 0 0 0 0]` |
| 6     | `[0 0 0 0 0 0 1 0 0 0]` |
| 7     | `[0 0 0 0 0 0 0 1 0 0]` |
| 8     | `[0 0 0 0 0 0 0 0 1 0]` |
| 9     | `[0 0 0 0 0 0 0 0 0 1]` |

---

#  Softmax Output

The output layer will use the **Softmax activation function**.

Softmax converts the network's raw output values into probabilities.

For example, after passing an image through the network, we might obtain:

```text
[0.01, 0.02, 0.01, 0.03, 0.01,
 0.87, 0.01, 0.01, 0.02, 0.01]
```

These values represent the model's estimated probabilities for digits:

```text
0 → 1%
1 → 2%
2 → 1%
3 → 3%
4 → 1%
5 → 87%
6 → 1%
7 → 1%
8 → 2%
9 → 1%
```

The probabilities sum to approximately:

```text
1.0
```

The class with the highest probability becomes the model's prediction.

In this example:

```text
Prediction = 5
Confidence = 87%
```

The goal is therefore not just to predict a class, but to obtain a meaningful **probability distribution over the 10 possible digits**.

---

#  Forward Propagation

During forward propagation, the input image moves through the network:

```text
784 Pixel Inputs
       ↓
Hidden Layer 1
       ↓
Hidden Layer 2
       ↓
10 Output Units
       ↓
Softmax
       ↓
10 Probabilities
```

The network produces a probability distribution that can be compared with the actual target.

For example:

```text
Target:

[0 0 0 0 0 1 0 0 0 0]


Prediction:

[0.01 0.01 0.02 0.01 0.01 0.89 0.01 0.01 0.02 0.01]
```

The model's prediction is strongly concentrated on digit `5`.

---

#  Loss Function

The predicted probability distribution is compared against the target distribution using a **loss function**.

For multi-class classification with a Softmax output, **categorical cross-entropy** is a common choice.

The loss measures how far the model's prediction is from the correct target.

The objective during training is to:

```text
Minimize Loss
```

A lower loss indicates that the model's predictions are becoming closer to the correct labels.

---

# ⚙️ Optimization

The neural network learns by adjusting its weights and biases.

The general training process is:

```text
Input Image
     ↓
Forward Propagation
     ↓
Prediction
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
Optimizer Updates Weights
     ↓
Repeat
```

An appropriate advanced optimizer can be selected to efficiently update the model parameters during training.

---

#  Backpropagation

Backpropagation allows the network to determine how much each parameter contributed to the prediction error.

The algorithm propagates the error backward through the network and calculates gradients for the model parameters.

The optimizer then uses these gradients to update the weights and biases.

Over many iterations, the model gradually learns patterns that distinguish one handwritten digit from another.

---

#  Dataset Splitting

The dataset should be divided into three subsets:

```text
MNIST Dataset
      │
      ├── Training Set
      │
      ├── Validation Set
      │
      └── Test Set
```

### Training Set

Used to learn the model's parameters.

### Validation Set

Used during development to monitor performance and tune model settings.

### Test Set

Used after training to evaluate the final model on unseen data.

Keeping the test set separate helps provide a more reliable measurement of how the trained model performs on new images.

---

#  Batch Training

Instead of processing the entire dataset at once, the training data can be divided into smaller groups called **batches**.

For example:

```text
Training Dataset
       ↓
Batch 1
Batch 2
Batch 3
...
Batch N
```

The **batch size** determines how many images are processed before the model performs a parameter update.

Choosing an appropriate batch size is part of the training configuration.

---

#  Data Preprocessing

Before training, the MNIST data should be prepared appropriately.

The general preprocessing pipeline is:

```text
Raw MNIST Images
       ↓
Convert to numerical arrays
       ↓
Normalize pixel values
       ↓
Flatten 28 × 28 images
       ↓
784-dimensional input vectors
       ↓
One-hot encode targets
       ↓
Split into Train / Validation / Test
```

Since the original pixel values range from `0` to `255`, they can be normalized to a smaller range such as:

```text
0 → 0.0
255 → 1.0
```

This generally makes optimization easier and provides better numerical scaling for the neural network.

---

#  MNIST Action Plan

The complete workflow for the project is:

### 1. Prepare the Data

Load the MNIST dataset and inspect its structure.

```text
70,000 images
28 × 28 pixels
10 classes
```

### 2. Preprocess the Data

* Normalize pixel values
* Flatten each image into 784 inputs
* One-hot encode the target labels
* Prepare the data for training

### 3. Create Dataset Splits

Create:

* Training dataset
* Validation dataset
* Test dataset

### 4. Select Batch Size

Choose an appropriate batch size for training.

### 5. Design the Neural Network

Build the architecture:

```text
784 Inputs
    ↓
Hidden Layer 1
    ↓
Hidden Layer 2
    ↓
10 Outputs
```

### 6. Choose Activation Functions

Use appropriate activation functions for the hidden layers and **Softmax** for the output layer.

### 7. Select an Optimizer

Choose an appropriate optimizer to update the network's parameters efficiently.

### 8. Select the Loss Function

Use an appropriate classification loss, such as **categorical cross-entropy**, for the Softmax output.

### 9. Train the Model

Allow the network to learn by repeatedly performing:

```text
Forward Pass
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Parameter Update
```

### 10. Validate at Each Epoch

Monitor the model's performance on the validation dataset after each epoch.

This helps track whether the model is learning effectively and whether its performance on unseen validation data is improving.

### 11. Test the Model

After training is complete, evaluate the final model on the test dataset.

The final evaluation should include metrics such as:

* Test loss
* Test accuracy
* Classification performance

---

#  Complete Pipeline

The entire MNIST classification pipeline can be summarized as:

```text
                    MNIST DATASET
                         │
                         ▼
                28 × 28 Grayscale
                       Images
                         │
                         ▼
                    Normalize
                  Pixel Values
                         │
                         ▼
                     Flatten
                   28 × 28 → 784
                         │
                         ▼
              ┌─────────────────────┐
              │    Input Layer      │
              │     784 Inputs      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Hidden Layer 1   │
              │  Linear + Nonlinear │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Hidden Layer 2   │
              │  Linear + Nonlinear │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Output Layer     │
              │     10 Outputs      │
              └──────────┬──────────┘
                         │
                         ▼
                      Softmax
                         │
                         ▼
               Probability for
                Digits 0 → 9
                         │
                         ▼
                  Calculate Loss
                         │
                         ▼
                  Backpropagation
                         │
                         ▼
                     Optimizer
                         │
                         ▼
                  Update Parameters
                         │
                         ▼
                  Repeat Training
                         │
                         ▼
                 Evaluate Accuracy
```

---

#  Objective

The ultimate objective is to build a model that can take a previously unseen handwritten digit image and correctly determine which number it represents.

For example:

```text
Input Image
     ↓
[Handwritten "5"]
     ↓
Neural Network
     ↓
Softmax
     ↓
[0.01, 0.01, 0.02, 0.01, 0.01,
 0.89, 0.01, 0.01, 0.02, 0.01]
     ↓
Predicted Digit: 5
```

The model should learn to distinguish between all **10 digit classes**, from `0` through `9`.

---

#  Learning Path

MNIST provides a useful progression for understanding neural networks:

```text
MNIST Dataset
      ↓
Data Preprocessing
      ↓
Fully Connected Neural Network
      ↓
Activation Functions
      ↓
Softmax Classification
      ↓
Loss & Optimization
      ↓
Backpropagation
      ↓
Validation & Testing
      ↓
Convolutional Neural Networks (CNNs)
```

Once the basic fully connected network is understood, the same classification problem can be revisited using a **CNN**, which can work directly with the spatial structure of the `28 × 28` images.

---

#  Key Takeaways

* MNIST contains **70,000 handwritten digit images** across **10 classes**.
* Each image is a **28 × 28 grayscale image**.
* Each image contains **784 pixels**.
* Each pixel acts as an input feature for the fully connected neural network.
* Pixel values range from **0 to 255** before normalization.
* Images are flattened from `28 × 28` into a **784-dimensional vector**.
* The network contains **two hidden layers**.
* The output layer contains **10 units**, one for each digit.
* Labels can be represented using **one-hot encoding**.
* **Softmax** converts the output into probabilities.
* A classification loss measures the difference between predictions and targets.
* **Backpropagation** calculates gradients.
* An **optimizer** updates the model parameters.
* Training, validation, and test sets are used for development and final evaluation.
* The final goal is accurate classification of unseen handwritten digits.

---

##  Final Goal

> **Build a deep neural network that learns from 784-pixel representations of handwritten digits and predicts the correct digit, from 0 to 9, with high accuracy.**

MNIST starts as a simple 10-class classification problem, but the concepts learned here form the foundation for much more advanced Deep Learning systems.
