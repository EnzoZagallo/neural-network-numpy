# Changelog

## v1.0.0 — 2026-06-07

Initial release of the neural network built from scratch with NumPy.

### Features

- Fully connected (Dense) layer with He weight initialization
- ReLU, Sigmoid, and Softmax activation functions with forward and backward passes
- Categorical Cross-Entropy loss function
- Combined Softmax + Cross-Entropy loss for numerical stability and faster backpropagation
- Sequential neural network model with training loop
- Vanilla Stochastic Gradient Descent (SGD) optimizer
- Spiral dataset generator with configurable classes, samples, and noise
- Training visualization: dataset plot, decision boundary, and loss/accuracy curves

### Project Structure

- `data.py` — spiral dataset generation
- `layers.py` — Dense layer implementation
- `activations.py` — ReLU, Sigmoid, Softmax
- `loss.py` — Cross-Entropy and combined Softmax + Cross-Entropy
- `model.py` — NeuralNetwork class orchestrating forward, backward, and update steps
- `train.py` — entry point for training and visualization

### Documentation

- README with motivation, architecture overview, and key insights
- Inline notes explaining the math behind each component
- Diagrams: neural network scheme, full epoch sequence, structure maps, and training results
