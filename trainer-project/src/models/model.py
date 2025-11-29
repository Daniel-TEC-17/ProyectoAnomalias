class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = self.initialize_weights(input_size, hidden_size)
        self.weights_hidden_output = self.initialize_weights(hidden_size, output_size)

    def initialize_weights(self, in_size, out_size):
        return np.random.randn(in_size, out_size) * 0.01

    def forward(self, x):
        self.hidden_layer = self.activation_function(np.dot(x, self.weights_input_hidden))
        self.output_layer = self.activation_function(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def backward(self, x, actual, learning_rate):
        output_error = self.output_layer - actual
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_layer * (1 - self.hidden_layer)

        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_layer.T, output_error)
        self.weights_input_hidden -= learning_rate * np.dot(x.T, hidden_error)

    def train(self, x, actual, learning_rate):
        predicted = self.forward(x)
        loss = self.compute_loss(predicted, actual)
        self.backward(x, actual, learning_rate)
        return loss