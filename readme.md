# Expected outcome of project

1. To prepare an [Artificial Neural Network](http://neuralnetworksanddeeplearning.com/) wrapper on this [code](https://github.com/mnielsen/neural-networks-and-deep-learning) for string classification (mainly useful for Computational linguitstics).
2. To prepare a log of wrongly identified classification and train further for their proper identificaion.


# Code modifications

The base code is [this](https://github.com/mnielsen/neural-networks-and-deep-learning/).

Brief overview of major modifications made by this wrapper are provided below.

1. Downloaded the data from http://sanskrit.uohyd.ernet.in/Corpus/SHMT/Samaas-Tagging/ in file input.txt (removed metadata).
2. Ran samAsa_array.php on this input.txt file to get three lists inputwords, outputwords and majorwords. (These lists are to be used by subsequent python programmes).
3. Copied the data of these three lists in preparation.py
4. In network2.py import heapq
7. In network2.py added functions unsolved, back_to_string, class_renamed, class_numbered and returnerror.
8. In network2.py SGD function, there are lines added to append unsolved_examples and they are stored in wrongdata.txt file for future manipulation.

-----

# Methodology for samAsa identification

The project started with the problem of samAsa identification for Sanskrit compounds. Its implementation details are given below.

1. According to this [paper](http://sanskrit.uohyd.ernet.in/Corpus/samAsaTagging-guide-lines.pdf), there are 5 major types and 55 minor types of samAsas.
2. We have scrapped the samAsa tagged data available online with [samAsa_array.php](https://github.com/drdhaval2785/SamaasaClassification/blob/master/samAsa_array.php) to get the data which we can use with the python programs.
3. The main code framework is the very neat and concise neural net code available [here](https://github.com/mnielsen/neural-networks-and-deep-learning/) (Text and explanation of the code are [here](http://neuralnetworksanddeeplearning.com/chap3.html))
4. The code there was primarily written to identify [MNIST digits classification](http://yann.lecun.com/exdb/mnist/). Input neuron had list with 784 entries ranging from 0-1. Output layer had 10 neurons corresponding to the number of digits. 
5. The problem is almost similar in case of samAsa classification. We have arbitrarily taken 50 as maximum number of letter in a samAsa. Therefore, the input layer has 50 neurons, each corresponding to a single letter. Output layer may have 5 or 55 neurons depending on whether we want to classify only major classes of samAsa or minor ones.
6. Input data is was devanagari with tagging in Roman transliteration. Therefore, the letters were converted to SLP1 format for ease of manipulation via php script [dev-slp.php](https://github.com/drdhaval2785/SamaasaClassification/blob/master/dev-slp.php).
7. After conversion, the next step for modification is the necessity of having the input values between 0.0 and 1.0. For that, a function sva() is written in [preparation.py](https://github.com/drdhaval2785/SamaasaClassification/blob/master/preparation.py). It converts the input letters to a number between 0.0-1.0 depending on encoding which we choose, just as we wanted.

-----

# Usage instructions

See codecopy.sh for some of the samples.

1. `preparation.py` has all the inputs and network parameters. This is the file where you would make changes to use the code for different usages.
2. Change inputwords and outputwords with your data. By default the database for samAsa classification is the input.
3. Set the various parameters of the network in preparation.py e.g. input_neuron, intermediate_neuron, epochs, mini_batch_size, eta, lmbda, howmany, mode.
4. Set whether you want to monitor the accuracies and costs in monitor_evaluation_cost, monitor_evaluation_accuracy, monitor_training_cost and monitor_training_accuracy. By default all of them are monitored.
5. Open your terminal.
6. Copy paste the code given below.
```
cd codes/SamaasaClassification
python -i
import preparation as p
training_data, test_data = p.load_data_wrapper(p.inputwords, p.outputwords, p.output_neuron, p.mode)
import network2
net = network2.Network([p.input_neuron, p.intermediate_neuron, p.output_neuron])
net.SGD(training_data, p.epochs, p.mini_batch_size, p.eta, p.all_class_types, p.mode, p.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=p.howmany)
net.save('networkstore.txt')
```

-----

# Explanations

1. `cd codes/SamaasaClassification` needs to be modified with your dictionary where the code is placed.
2. `python -i` opens the python terminal and ensures that the next code is executed in python shell.
3. `import preparation as p` imports the module preparation
4. `training_data, test_data = p.load_data_wrapper(p.inputwords, p.outputwords, p.output_neuron, p.mode)` loads the data from the inputwords, outputwords and majorwords to our network for use.
5. `import network2` imports the network2.py which has the structure and code for self learning by ANN.
6. `net = network2.Network([p.input_neuron, p.intermediate_neuron, p.output_neuron])` initialises our network named `net`.
7. `net.SGD(training_data, p.epochs, p.mini_batch_size, p.eta, p.all_class_types, p.mode, p.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=p.howmany)` - This is the core part of the code which actually learns from the input data. See network2.py for its parameters details. You may like / need to modify various parameters of this network to ensure that your ANN learns well.
8. `net.save('networkstore.txt')` saves the network details to `networkstore.txt` file, so that we don't have to repeat the whole training again and again.

-----

# Reloading a learnt ANN

```
cd codes/SamaasaClassification
python -i
import preparation
training_data, test_data = p.load_data_wrapper(p.inputwords,p.outputwords, p.output_neuron, p.mode)
import network2
net = network2.load('networkstore.txt')
net.SGD(training_data, p.epochs, p.mini_batch_size, p.eta, p.all_class_types, p.mode, p.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=p.howmany)
net.save('networkstore.txt')
```

The explanation remains the same.

-----

# Outputs

1. The console would show the accuracy and cost after every epoch.
2. The log file would store the following details: sizes, epochs, mini batch size, eta, lmbda, max training accuracy, max evaluation accuracy, min training cost, min evaluation cost, traininig accuracy, evaluation accuracy, weights, biases, unsolved examples, time of the training.
3. `networkstore.txt` file would store the network details for future reload.
4. `wrongdata.txt` file would store the strings which were wrongly classified by ANN.

-----

# Todo

1. Make it usable across UTF8 character range

-----

# Results in samAsa classification

##### Classification in 5 major samAsa classes


##### Classification in 55 minor samAsa classes

