"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys
from decimal import Decimal # Added by Dhaval
import heapq # Added by Dhaval
import datetime # Added by Dhaval

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime_vec(z)


class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network():

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

	# Dhaval added parameters classtypes, mode and items
    def SGD(self, training_data, epochs, mini_batch_size, eta, classtypes, mode,
            lmbda = 0.0, 
            evaluation_data=None, 
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False, 
            monitor_training_accuracy=False,
            items=1):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        output_neuron = self.sizes[2]
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        f = open("wrongdata.txt", "w")
        unsolved_examples = []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda, output_neuron)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True, items=items)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
                unsolved_examples.append(self.unsolved(training_data, classtypes, mode, convert=True)) # Added to log the wrongly identified data
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, output_neuron, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data)
                unsolved_examples.append(self.unsolved(evaluation_data, classtypes, mode, convert=False)) # Added to log the wrongly identified data
            print (max(training_accuracy)*100.0)/len(training_data) # Print maximum accuracy across epochs
            print (max(evaluation_accuracy)*100.0)/len(evaluation_data) # print maximum accuracy across epochs
            print
        wrongs = 0
        for mem in unsolved_examples:
			wrongs = wrongs + len(mem)
        print "Total wrongly identified samAsas are {}".format(wrongs) # Print total number of wrong identification in all epochs.
        final_examples = self.deflate(unsolved_examples) # Removed duplicates
        print "Unique wrongly identified samAsas are {}".format(len(final_examples)) # Print total unique wrongly identified data in all epochs.
        json.dump(final_examples, f) # Save the wrong data in file wrongdata.txt for further manipulation, if needed.
        f.close()
        self.logonce(epochs, mini_batch_size, eta, lmbda, training_accuracy, evaluation_accuracy, training_cost, evaluation_cost, final_examples, training_data, evaluation_data) # Logging all necessary information along for reconstruction of network, accuracies, costs etc with wrong identified data
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

	# No change
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

	# No change
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

	# Added the parameter items. (default is items=1 i.e. the most activated is matched. If items=2, then the two most activated are compared against the correct answer and if found matching, it is considered correct. Equivalent to "The answer is in 'n' most activated neurons")
    def accuracy(self, data, convert=False, items=1):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.  

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(map(self.feedforward(x).tolist().index, heapq.nlargest(items, self.feedforward(x).tolist())), np.argmax(y)) for (x, y) in data]
        else:
            results = [(map(self.feedforward(x).tolist().index, heapq.nlargest(items, self.feedforward(x).tolist())), y)
                        for (x, y) in data]	
        counter = 0
        for (x, y) in results:
            if y in x:
                counter = counter + 1
        return counter

	# Added by Dhaval to append the data of unsolved examples.
    def unsolved(self, data, classtypes, mode, convert=False, items=1):

		#results = [(np.argmax(self.feedforward(x)), np.argmax(y), x, y) for (x, y) in data]
        if convert:
            results = [(map(self.feedforward(x).tolist().index, heapq.nlargest(items, self.feedforward(x).tolist())), np.argmax(y), x, y) for (x, y) in data]
        else:
            results = [(map(self.feedforward(x).tolist().index, heapq.nlargest(items, self.feedforward(x).tolist())), y, x, y)
                        for (x, y) in data]

        unsol = []
        for (a, b, c, d) in results:
            if b == a:
                pass
            elif b in a:
                pass
            else:
                unsol.append(self.returnerror(self.class_renamed(np.argmax(self.feedforward(c)), classtypes), self.class_renamed(np.argmax(d), classtypes), self.back_to_string(c.tolist(), mode) ))
        return unsol

	# Added by Dhaval. It is reverse of sva in preparation.py. Converts the numbers back to human readable format.
    def back_to_string(self, x, mode):
        val = ''
        if mode == "ASCII":
			letters = [chr(p) for p in range(32, 128)]
			values = [(ord(q)-32.0)/96.0 for q in letters]
			letters[0] = ''
			for i in range(len(x)):
				val = val+letters[values.index(x[i][0])]
				val.replace(' ','')
			return val
        elif mode == "Sanskrit":
			letters = ["a","A","i","I","u","U","f","F","x","X","e","E","o","O","k","K","g","G","N","c","C","j","J","Y","w","W","q","Q","R","t","T","d","D","n","p","P","b","B","m","y","r","l","v","S","z","s","h","M","!","H","-",""]
			values = [0.019230769230769,0.038461538461538,0.057692307692308,0.076923076923077,0.096153846153846,0.11538461538462,0.13461538461538,0.15384615384615,0.17307692307692,0.19230769230769,0.21153846153846,0.23076923076923,0.25,0.26923076923077,0.28846153846154,0.30769230769231,0.32692307692308,0.34615384615385,0.36538461538462,0.38461538461538,0.40384615384615,0.42307692307692,0.44230769230769,0.46153846153846,0.48076923076923,0.5,0.51923076923077,0.53846153846154,0.55769230769231,0.57692307692308,0.59615384615385,0.61538461538462,0.63461538461538,0.65384615384615,0.67307692307692,0.69230769230769,0.71153846153846,0.73076923076923,0.75,0.76923076923077,0.78846153846154,0.80769230769231,0.82692307692308,0.84615384615385,0.86538461538462,0.88461538461538,0.90384615384615,0.92307692307692,0.94230769230769,0.96153846153846,0.98076923076923,0.0]
			val = ''
			for i in range(len(x)):
				val = val+letters[values.index(x[i][0])]
			return val
    
    # Added by Dhaval. It returns the classname from the number.
    def class_renamed(self, x, classtypes):
        return classtypes[x]

	# Added by Dhaval. returns the data in format amenable to json.
    def returnerror(self, a, b, c):
        data = {"machine_answer": a,
                "correct_answer": b,
                "input_entered": c,}
        return data

	# Added by Dhaval. Flattens a two layered array
    def deflate(self, a):
		"""flattening a two layered array"""
		data = []
		for mem1 in a:
			for mem2 in mem1:
			    if mem2 not in data:
				    data.append(mem2)
		return data
    
	# Added parameter output_neuron to make it generalised.
    def total_cost(self, data, lmbda, output_neuron, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y, output_neuron)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

	# No change. Saves the neural network.
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
	
	# Preparing a log of the network parameters, accuracies, costs and wrongly identified members for future reconstruction. Maybe we should enter input and outputs as well.
    def logonce(self, epochs, mini_batch_size, eta, lmbda, training_accuracy, evaluation_accuracy, training_cost, evaluation_cost, final_examples, training_data, evaluation_data):
		""" Create a log of the network training with necessary details"""
		f = open("log.txt","a")
		data = {"sizes": self.sizes,
				"epochs": epochs,
				"mini_batch_size": mini_batch_size,
				"eta": eta,
				"lambda": lmbda,
				"max_training_accuracy": (max(training_accuracy)*100.0)/len(training_data),
				"max_evaluation_accuracy": (max(evaluation_accuracy)*100.0)/len(evaluation_data),
				"min_training_cost": min(training_cost),
				"min_evaluation_cost": min(evaluation_cost),
				"traininig_accuracy": training_accuracy,
				"evaluation_accuracy": evaluation_accuracy,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
				"unsolved_examples": final_examples,
			    "time": str(datetime.datetime.now())}
		json.dump(data, f)
		f.write("\n" + "--------------------" + "\n")
		f.close()
		

#### Loading a Network
# No change.
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"], cost=CrossEntropyCost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
# Dhaval added a parameter output_neuron for generalization.
def vectorized_result(j, output_neuron):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((output_neuron, 1))
    e[j] = 1.0
    return e

# No change
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

# No change
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


sigmoid_prime_vec = np.vectorize(sigmoid_prime)
