import numpy as np


class Dense:
    """Fully connected dense layer. Every input connects to every neuron.
    Used as the fundamental building block of feed-forward neural networks.
    """

    def __init__(self, n_inputs, n_neurons):
        """Initializing the value for weights using a vector of n_inputs rows
        by n_neurons columns and multiplying it by sqrt(2 / n_inputs) to set
        the variance to 2/n_inputs, keeping activation magnitudes stable
        across layers when using ReLU. We choose 2 instead of 1 because we
        are using the ReLU activation function, which sets all the negative
        values to zero, then halving the variance of the output.

        We create for the biases a vector of zeros with 1 row and n_neurons
        columns, which will be added to the output of the layer after the dot
        product of inputs and weights.
        """
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(
            2.0 / n_inputs
        )
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """For the forward method, we cache the inputs and then multiply them
        with the weights, adding the biases to the result of the dot product,
        thus getting the outputs necessary for further evaluating the loss of
        each batch of inputs and their corresponding weights and biases.
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        """Calculate the gradients for the weights, the biases and the inputs
        so that they can be used for the backpropagation process, which is the
        process of updating the weights and biases through each layer.

        The gradients of the weights are attained through the multiplication
        of the transposed inputs with the derivative values, getting the
        proper weight for each derivative. While the weights are multiplied,
        the biases are added to the outputs, thus sufficing to set the bias
        gradients as the sum of the dvalues. We preserve the shape of the
        biases to prevent collapsing of the dimensions, which is 1 row and
        n_neurons columns, using the keepdims parameter. We calculate the
        gradients of the dinputs by multiplying the dvalues with the
        transposed weights, giving a gradient of shape (batch, n_inputs). It
        measures how much each input value contributed to the loss, translated
        back through the weights. This is returned to the previous layer as
        its own dvalues to continue the backward chain. dweights and dbiases
        are stored as attributes for the optimizer to collect and update
        parameters after the full backward pass.

        dweights  =  dL/dW   derivative of loss with respect to weights
        dbiases   =  dL/db   derivative of loss with respect to biases
        dinputs   =  dL/dX   derivative of loss with respect to inputs
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
