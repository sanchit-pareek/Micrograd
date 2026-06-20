# Micrograd

A tiny scalar-valued autograd engine (automatic differentiation) and a small neural network library built on top of it with a PyTorch-like API.

This project is designed to understand how automatic differentiation and backpropagation work under the hood.

## Overview

Micrograd implements:

- **Autograd Engine**: A minimal reverse-mode automatic differentiation system (~100 lines)
- **Neural Network Library**: A lightweight NN library with a PyTorch-like interface (~50 lines)
- **Scalar-based Operations**: All computations are performed on scalar values, breaking down each neuron operation into individual arithmetic operations

This approach makes the entire system transparent and understandable, while still being powerful enough to build and train complete deep neural networks for binary classification tasks.

## Features

- ✨ Simple, readable implementation of backpropagation
- 📊 Support for various operations: addition, multiplication, exponentiation, ReLU activation
- 🧠 Build multi-layer neural networks (MLPs)
- 📈 Binary classification capability
- 🎓 Fully educational - understand every line of code

## Getting Started

### Installation

```bash
pip install micrograd
```

Or install from source:

```bash
git clone https://github.com/sanchit-pareek/Micrograd.git
cd Micrograd
pip install -e .
```

### Basic Usage Example

```python
from micrograd.engine import Value

# Create some values
a = Value(-4.0)
b = Value(2.0)

# Build a computation graph
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

# Forward pass result
print(f'{g.data:.4f}')  # prints 24.7041

# Backward pass - compute gradients
g.backward()

# Access gradients
print(f'{a.grad:.4f}')  # dg/da
print(f'{b.grad:.4f}')  # dg/db
```

## Notebooks

This repository includes educational Jupyter notebooks:

- **`micrograd_lecture_first_half_roughly.ipynb`** - Introduction to the autograd engine and fundamentals
- **`micrograd_lecture_second_half_roughly.ipynb`** - Neural networks, training, and practical examples

Open these notebooks to see detailed explanations and interactive examples.

## Key Concepts

### Value Object
The core of micrograd is the `Value` class, which wraps a scalar and tracks its dependencies to compute gradients.

### Computation Graph
Every operation creates a directed acyclic graph (DAG) of operations, allowing us to compute gradients via backpropagation.

### Backpropagation
The `backward()` method traverses the computation graph in reverse topological order, applying the chain rule to compute gradients for all values.

## Example: Binary Classification

With micrograd, you can build and train a 2-layer neural network (MLP) for binary classification:

```python
from micrograd import nn
from micrograd.engine import Value

# Create a 2-layer network
model = nn.MLP(2, [16, 16, 1])

# Simple training loop
for epoch in range(100):
    # Forward pass
    logits = [model(x) for x in X]
    
    # Loss computation (e.g., MSE)
    loss = sum((pred - y)**2 for pred, y in zip(logits, Y))
    
    # Backward pass
    loss.backward()
    
    # Update weights (simple SGD)
    for p in model.parameters():
        p.data -= 0.01 * p.grad
```

## Project Structure

```
micrograd/
├── README.md
├── micrograd_lecture_first_half_roughly.ipynb
├── micrograd_lecture_second_half_roughly.ipynb
```

## Learning Resources

- Start with the lecture notebooks to understand the concepts
- Read through the implementation to see how it all fits together
- Modify the examples and experiment with different architectures
- Compare with PyTorch to understand the differences

## License

MIT License - See LICENSE file for details

## References

- Original Micrograd: [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- By [Andrej Karpathy](https://github.com/karpathy)

## Educational Purpose

This project is designed for educational purposes. It's an excellent resource for understanding:
- How automatic differentiation works
- Backpropagation and the chain rule
- Neural network fundamentals
- Building neural networks from scratch

---

Happy learning! Feel free to experiment and modify the code to deepen your understanding of deep learning fundamentals.
