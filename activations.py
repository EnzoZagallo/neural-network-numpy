import numpy as np


class ReLU:
    """Rectified Linear Unit activation function."""

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs


class Sigmoid:
    """Sigmoid activation function. We use clipping to prevent overflow in exp()."""

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)
        return self.dinputs


class Softmax:
    """Softmax activation function for output layer (multi-class). Using the jacobian to assess the gradient and influence of each output on the others."""

    def forward(self, inputs):
        # Subtract max for numerical stability and divide by sum to get probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            self.dinputs[i] = np.dot(jacobian, single_dvalues)
        return self.dinputs
