# Expected outcome of project

1. To prepare an Artificial Neural Network wrapper for string classification (mainly useful for Computational linguitstics).
2. To prepare a log of wrongly identified classification and train further for their proper identificaion.


# Code modifications

The base code is that of https://github.com/mnielsen/neural-networks-and-deep-learning/
Brief overview of major modifications are provided below

1. Downloaded the data from http://sanskrit.uohyd.ernet.in/Corpus/SHMT/Samaas-Tagging/ in file input.txt (removed metadata).
2. Ran samAsa\_array.php on this input.txt file to get three lists inputwords, outputwords and majorwords. (These lists are to be used by subsequent python programmes).
3. Copied the data of these three lists in preparation.py
4. In network2.py import heapq
7. In network2.py added functions unsolved, back\_to\_string, class\_renamed, class\_numbered and returnerror.
8. In network2.py SGD function, there are lines added to append unsolved\_examples and they are stored in wrongdata.txt file for future manipulation.

-----

# Methodology for samAsa identification

The project started with the problem of samAsa identification for Sanskrit compounds. Its implementation details are given below.

1. According to the paper http://sanskrit.uohyd.ernet.in/Corpus/samAsaTagging-guide-lines.pdf, there are 5 major types and 55 minor types of samAsas.
2. We have scrapped the samAsa tagged data available online with samAsa\_array.php to get the data which we can use with the python programs.
3. The main code framework is the very neat and concise neural net code available at https://github.com/mnielsen/neural-networks-and-deep-learning/ (Text and explanation of the code on http://neuralnetworksanddeeplearning.com/chap3.html)
4. The code there was primarily written to identify MNIST digits classification. Input neuron had list with 784 entries ranging from 0-1. Output layer had 10 neurons corresponding to the number of digits. 
5. The problem is almost similar in case of samAsa classification. We have arbitrarily taken 50 as maximum number of letter in a samAsa. Therefore, the input layer has 50 neurons, each corresponding to a single letter. Output layer may have 55 or 5 neurons depending on whether we want to classify only major classes of samAsa or minor ones.
6. Input data is in devanagari with tagging in Roman transliteration. Therefore, the letters were converted to SLP1 format for ease of manipulation via php script dev-slp.php.
7. After conversion, the next step for modification is the necessity of having the input values between 0.0 and 1.0. For that, a function sva() is written in preparation.py. It assigns the value of index/52 to each letter. It converts the input letters to a number between 0.0-1.0, just as we wanted.

-----

# Usage instructions

See codecopy.sh for some of the samples.

1. Open your terminal.
2. Copy paste the code given below.
```
cd codes/SamaasaClassification
python -i
import preparation as p
training\_data, test\_data = p.load\_data\_wrapper(p.inputwords, p.outputwords, p.output\_neuron, p.mode)
import network2
net = network2.Network([p.input\_neuron, p.intermediate\_neuron, p.output\_neuron])
net.SGD(training\_data, p.epochs, p.mini\_batch\_size, p.eta, p.all\_class\_types, p.mode, p.lmbda, evaluation\_data=test\_data, monitor\_evaluation\_accuracy=True, monitor\_evaluation\_cost=True, monitor\_training\_accuracy=True, monitor\_training\_cost=True, items=p.howmany)
net.save('networkstore.txt')
```

-----

# Explanations

1. `cd codes/SamaasaClassification` needs to be modified with your dictionary where the code is placed.
2. `python -i` opens the python terminal and ensures that the next code is executed in python shell.
3. `import preparation as p` imports the module preparation
4. `training\_data, test\_data = p.load\_data\_wrapper(p.inputwords, p.outputwords, p.output\_neuron, p.mode)` loads the data from the inputwords, outputwords and majorwords to our network for use.
5. `import network2` imports the network2.py which has the structure and code for self learning by ANN.
6. `net = network2.Network([p.input\_neuron, p.intermediate\_neuron, p.output\_neuron])` initialises our network named `net`.
7. `net.SGD(training\_data, p.epochs, p.mini\_batch\_size, p.eta, p.all\_class\_types, p.mode, p.lmbda, evaluation\_data=test\_data, monitor\_evaluation\_accuracy=True, monitor\_evaluation\_cost=True, monitor\_training\_accuracy=True, monitor\_training\_cost=True, items=p.howmany)` - This is the core part of the code which actually learns from the input data. See network2.py for its parameters details. You may like / need to modify various parameters of this network to ensure that your ANN learns well.
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

-----

# Todo

1. Make it usable across UTF8 character range


