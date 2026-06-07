import numpy as np

from layers import Dense
from loss import SoftmaxCrossEntropyLoss


class NeuralNetwork:
    """Sequential neural network built from scratch with NumPy."""

    def __init__(self):
        self.layers = []          # (Dense, activation) pairs
        self.loss_fn = SoftmaxCrossEntropyLoss()
        self.training_history = {"loss": [], "accuracy": []}

    def add(self, layer, activation):
        """Add a dense layer followed by its activation function. Dense layers produces raw numbers and activation creater non-linearity."""
        self.layers.append((layer, activation))

    def forward(self, X):
        """Forward pass through all layers."""
        output = X
        for layer, activation in self.layers:
            output = layer.forward(output)
            output = activation.forward(output)
        return output

    def backward(self, targets):
        """Backward pass — compute gradients for all layers."""
        # Combined softmax + cross-entropy backward (skips last activation)
        grad = self.loss_fn.backward(targets)

        # Backprop through layers in reverse (skip last activation's backward), for 3 layers network, we dont consider the first Layer, as i=2 is not in the conditional.
        for i in reversed(range(len(self.layers))):
            layer, activation = self.layers[i]
            if i < len(self.layers) - 1:
                grad = activation.backward(grad)
            grad = layer.backward(grad)

    def update_params(self, learning_rate):
        """Update weights and biases using vanilla Stochastis Gradient Descent, simples possible optimizer."""
        for layer, _ in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases

    def compute_loss(self, output, targets):
        """Compute the loss using the combined softmax + cross-entropy."""
        return self.loss_fn.forward(output, targets)

    def compute_accuracy(self, targets):
        """Compute classification accuracy from the last forward pass.
        We use the boolean values of the comparison to compute the mean accuracy."""
        predictions = np.argmax(self.loss_fn.output, axis=1)
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        return np.mean(predictions == targets)

    def train(self, X, y, epochs=1000, learning_rate=1.0, print_every=100):
        """Full training loop combining each step: forward, loss, backward, and update."""
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)

            # Loss (uses raw pre-softmax from last layer — we need to pass
            # the pre-activation output, so we grab it directly)
            last_layer, _ = self.layers[-1]
            loss = self.compute_loss(last_layer.output, y)
            accuracy = self.compute_accuracy(y)

            self.training_history["loss"].append(loss)
            self.training_history["accuracy"].append(accuracy)

            # Backward
            self.backward(y)

            # Update
            self.update_params(learning_rate)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:>5d}/{epochs}  |  "
                    f"Loss: {loss:.4f}  |  Accuracy: {accuracy:.4f}"
                )

        print("\nTraining complete.")
        print(f"Final loss: {loss:.4f}  |  Final accuracy: {accuracy:.4f}")
