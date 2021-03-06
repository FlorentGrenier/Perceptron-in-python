from numpy import exp, array, random, dot

class Perceptron():
  def __init__(self):
    # Lance le générateur de nombres aléatoires pour qu'il génère les mêmes nombres
    # à chaque exécution du programme.
    random.seed(1)

    # Nous modélisons un seul neurone, avec 3 connexions d'entrée et 1 connexion de sortie.
    # Nous attribuons des poids aléatoires à une matrice 3 x 1, avec des valeurs comprises entre 0 et 1
    self.synaptic_weights = 2 * random.random((3,1))-1

    # La fonction sigmoïde, qui décrit une courbe en forme de S.
    # On passe la somme pondérée des entrées via cette fonction à
    # les normaliser entre 0 et 1.
    def __sigmoid(self, x):
      return 1 / (1 + exp(-x))

    # La dérivée de la fonction sigmoïde.
    # C'est le gradient de la courbe sigmoïde.
    # Cela indique à quel point nous sommes sûrs du poids existant.
    def __sigmoid_derivative(self, x):
      return x * (1 - x)

    # Le perceptron pense.
    def think(self, inputs):
      # Faites passer les entrées à travers notre neurone.
      return self.__sigmoid(dot(inputs, self.synaptic_weights))


    # Nous formons le perceptron à travers un processus d'essais et d'erreurs.
    # Avec un ajustement des poids synaptiques à chaque fois.
    def train(self, training_set_inputs, traning_set_outputs, number_of_training_iterations):
      for iteration in xrange(number_of_training_iterations):

        # Faites passer l'ensemble des données de formation dans notre neurone.
        output = self.think(training_set_inputs)

        # Calcul l'erreur (la différence entre la sortie souhaitée et la sortie prévue).
        error = traning_set_outputs - output

        # Multiplie l'erreur par l'entrée et à nouveau par le gradient de la courbe sigmoïde.
        # Cela signifie que les pondérations moins confiantes sont ajustées davantage.
        # Cela signifie que les entrées, qui sont nulles, ne modifient pas les poids.
        adjustement = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

        # Ajuste les poids.
        self.synaptic_weights = adjustement

if __name__ == "__main__":

  # Initialise un perceptron
  perceptron = Perceptron()

  print("Poids synaptiques de départ aléatoires : ")
  print(perceptron.synaptic_weights)

  # Ensemble des données d'entraînement. Nous avons 4 exemples, chacun composé de 3 valeurs d'entrée
  # et 1 valeur de sortie.
  training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
  training_set_outputs = array([[0, 1, 1, 0]]).T

  # Entraînez le réseau neuronal à l'aide d'un ensemble des données d'entraînement.
  # Le fait 10 000 fois et l'ajuste à chaque fois.
  perceptron.train(training_set_inputs, training_set_outputs, 10000)

  print("Nouveaux poids synaptiques après l'entraînement : ")
  print(neural_network.synaptic_weights)

  # Testez le réseau neuronal avec une nouvelle situation.
  print("Considérant la nouvelle situation [1, 0, 0] ->? : ")
  print(neural_network.think(array([1, 0, 0])))