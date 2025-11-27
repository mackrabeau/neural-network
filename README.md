# Simple Educational Neural Network (NumPy)

A minimal, from-scratch neural network implementation in Python using NumPy and SciPy for core operations. This repository is intended for learning and experimentation — it implements basic layers (dense, convolutional), activation functions, loss functions, and small training loops without using high-level deep learning frameworks (except to download datasets).

This project was developed for learning and experimentation and includes code adapted from an educational YouTube tutorial. Inspiration and some implementation ideas were taken from the tutorial at:

https://youtu.be/Lakz2MoHy6o?si=z0mRq38t3pmzf29Y

## Contents

- `activation.py` / `activations.py` - Activation layer and common activations (Sigmoid, Tanh).
- `layer.py` - Base `Layer` interface.
- `dense.py` - Fully connected (`Dense`) layer implementation.
- `convolutional.py` - Simple convolutional layer using `scipy.signal`.
- `reshape.py` - Utility layer to reshape tensors between layers.
- `losses.py` - Implemented loss functions (MSE, binary cross-entropy) and their derivatives.
- `network.py` - `train` and `predict` helper functions (simple training loop).
- `mnist.py` - Example: train a small dense network on MNIST (CPU-friendly subset).
- `mnist_conv.py` - Example: small convolutional network on a binary MNIST subset (0 vs 1).
- `xor.py` - (If present) small example for learning XOR.

## Requirements

- Python 3.8+ (developed with Python 3.x)
- NumPy
- SciPy
- TensorFlow (only to load the MNIST dataset via `tensorflow.keras.datasets`)

## Usage

The examples are small scripts that train on limited subsets so they can run on CPU.

- Train a small dense MNIST classifier (uses MSE + Tanh):

```bash
python mnist.py
```

- Train/test the small convolutional example (binary 0 vs 1):

```bash
python mnist_conv.py
```
## How it works 

- Each layer implements `forward(input)` and `backward(output_gradient, learning_rate)`.
- The `train` loop in `network.py` performs forward passes for each sample, computes loss, backpropagates gradients through each layer (in reverse), and updates parameters using simple gradient descent.

