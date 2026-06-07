import numpy as np


class CategoricalCrossEntropy:
    """Categorical cross-entropy loss for multi-class classification.
    Predictions being the output from softmax activation.
    We clip using a non significant number, 1e-7 to prevent infinities
    For cross entropy loss, we only care if the predicted class is correct, 
    Logarithm punishes wrong predictions exponetially with -log(correct_confidence)
    We calculate the mean loss across all samples to normalize the value for the batch size."""
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        samples = len(predictions)

        # Clip to prevent log(0)
        clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        # If targets are one-hot encoded
        if targets.ndim == 2:
            correct_confidences = np.sum(clipped * targets, axis=1)
        # If targets are class indices, telling exactly which class is correct by index of rows and columns
        else:
            correct_confidences = clipped[range(samples), targets]

        loss = -np.log(correct_confidences)
        return np.mean(loss)

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        num_classes = dvalues.shape[1]

        # Convert class indices to one-hot if needed
        if targets.ndim == 1:
            targets = np.eye(num_classes)[targets]

        # Gradient of loss with respect to softmax output, returning the gradient of the loss for each parameter
        self.dinputs = -targets / np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Normalize by batch size
        self.dinputs = self.dinputs / samples
        return self.dinputs


class SoftmaxCrossEntropyLoss:
    """Combined Softmax + Cross-Entropy for faster & more stable backprop.
    We join both the softmax activation and the cross-entropy loss into a single class to 
    optimize the backward pass."""

    def forward(self, inputs, targets):
        # Softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Cross-entropy loss
        samples = len(inputs)
        clipped = np.clip(self.output, 1e-7, 1 - 1e-7)

        if targets.ndim == 1:
            correct_confidences = clipped[range(samples), targets]
        else:
            correct_confidences = np.sum(clipped * targets, axis=1)

        return -np.mean(np.log(correct_confidences))

    def backward(self, targets):
        #Origin of the backward chain.
        samples = len(self.output)
        self.dinputs = self.output.copy()

        if targets.ndim == 1:
            self.dinputs[range(samples), targets] -= 1
        else:
            self.dinputs -= targets

        self.dinputs = self.dinputs / samples
        return self.dinputs
