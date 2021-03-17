from numpy import exp, array, random, dot

class Perceptron():
  def __init__(self):
    # Start the random number generator to generate the same numbers
    # each time the program is run.
    random.seed(1)

    # We model a single neuron, with 3 input connections and 1 output connection.
    # We assign random weights to a 3 x 1 matrix, with values ​​between 0 and 1.
    self.synaptic_weights = 2 * random.random((3,1))-1

    # The sigmoid function, which describes an S-shaped curve.
    # We pass the weighted sum of the inputs via this function to
    # normalize them between 0 and 1.
    def __sigmoid(self, x):
      return 1 / (1 + exp(-x))

    # The derivative of the sigmoid function.
    # This is the gradient of the sigmoid curve.
    # This indicates how sure we are of the existing weight.
    def __sigmoid_derivative(self, x):
      return x * (1 - x)

    # The perceptron thinks.
    def think(self, inputs):
      # Pass the inputs through the neuron.
      return self.__sigmoid(dot(inputs, self.synaptic_weights))


    # We train the perceptron through a process of trial and error.
    # With adjustment of synaptic weights each time.
    def train(self, training_set_inputs, traning_set_outputs, number_of_training_iterations):
      for iteration in xrange(number_of_training_iterations):

        # Pass all the training data through our neuron.
        output = self.think(training_set_inputs)

        # Calculate the error (the difference between the desired output and the expected output).
        error = traning_set_outputs - output

        # Multiply the error by the input and again by the gradient of the sigmoid curve.
        # This means less confident weights are adjusted further.
        # This means that the entries, which are zero, do not modify the weights.
        adjustement = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

        # Adjust the weights.
        self.synaptic_weights = adjustement

if __name__ == "__main__":

  # Initialize a perceptron
  perceptron = Perceptron()

  print("Random starting synaptic weights : ")
  print(perceptron.synaptic_weights)

  # All training data. We have 4 examples, each made up of 3 input values
  # and 1 output value.
  training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
  training_set_outputs = array([[0, 1, 1, 0]]).T

  # Train the neural network using a set of training data.
  # Do it 10,000 times and adjust it each time.
  perceptron.train(training_set_inputs, training_set_outputs, 10000)

  print("New synaptic weights after training : ")
  print(neural_network.synaptic_weights)

 # Test the neural network with a new situation.
  print("Considering the new situation [1, 0, 0] ->? : ")
  print(neural_network.think(array([1, 0, 0])))