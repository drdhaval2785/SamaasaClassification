# Expected outcome of project

1. To prepare an Artificial Neural Network for classifying the samAsas of Sanskrit language
2. To prepare a log of wrongly identified samAsas and train further for their proper identificaion.


# Code modifications

The base code is that of https://github.com/mnielsen/neural-networks-and-deep-learning/
Brief overview of major modifications are provided below

1. Downloaded the data from http://sanskrit.uohyd.ernet.in/Corpus/SHMT/Samaas-Tagging/ in file input.txt (removed metadata).
2. Ran samAsa_array.php on this input.txt file to get three lists inputwords, outputwords and majorwords. (These lists are to be used by subsequent python programmes).
3. Copied the data of these three lists in trialcode.py
4. In trialcode.py the functions loadable_data and loadable_data_major may need modification in numbers depending on the input list size.
5. In trialcode.py, changed the value of parameter j of function vectorized_result and vectorized_result_major depending on the final output classification e.g. for full samAsa range 55 and for major range 5.
6. In network2.py import heapq
7. In network2.py added functions unsolved, back_to_string, samAsa_renamed, samAsa_numbered and returnerror.
8. In network2.py SGD function, there are lines added to append unsolved_examples and they are stored in wrongdata.txt file for future manipulation.

-----

# Methodology
1. According to the paper http://sanskrit.uohyd.ernet.in/Corpus/samAsaTagging-guide-lines.pdf, there are 5 major types and 55 minor types of samAsas.
2. We have scrapped the samAsa tagged data available online with samAsa_array.php to get the data which we can use with the python programs
3. The main code framework is the very neat and concise neural net code available at https://github.com/mnielsen/neural-networks-and-deep-learning/ (Text and explanation of the code on http://neuralnetworksanddeeplearning.com/chap3.html)
4. The code there was primarily written to identify MNIST digits classification. Input neuron had list with 784 entries ranging from 0-1. Output layer had 10 neurons corresponding to the number of digits. 
5. The problem is almost similar in case of samAsa classification. We have arbitrarily taken 50 as maximum number of letter in a samAsa. Therefore, the input layer has 50 neurons. Output layer may have 55 or 5 neurons depending on whether we want to classify only major classes of samAsa or minor ones.
6. Input data is in devanagari with tagging in Roman transliteration. Therefore, the letters were converted to SLP1 format for ease of manipulation.
7. After conversion, the next step for modification is the necessity of having the input values between 0.0 and 1.0. For that, a function sva() is written in trialcode.py. It assigns the value of index/52 to each letter. It converts the input letters to a number between 0.0-1.0, just as we wanted. If the input convention needs more than 52 letters, the denominator needs to be changed to a suitably higher number. 

-----

# Usage instructions
See codecopy.sh for some of the samples.

1. Open your terminal.
2. Copy paste the code given below.
```
cd codes/gittest/anndev/src
python -i
import trialcode
training_data, validation_data, test_data = trialcode.load_data_wrapper_major(trialcode.inputwords,trialcode.majorwords)
import network2
net = network2.Network([50, 5])
net.SGD(training_data, 1, 10, 1.5,lmbda = 0.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)
net.save('networkstore.txt')
```
-----

# Explanations

1. `cd codes/gittest/anndev/src` needs to be modified with your dictionary where the code is placed.
2. `python -i` opens the python terminal and ensures that the next code is executed in python shell.
3. `import trialcode` imports the module trialcode
4. `training_data, validation_data, test_data = trialcode.load_data_wrapper_major(trialcode.inputwords,trialcode.majorwords)` loads the data from the inputwords, outputwords and majorwords to our network for use.
5. `import network2` imports the network2.py which has the structure and code for self learning by ANN.
6. `net = network2.Network([50, 5])` initialises our network named `net` with 50 input neuron and 5 output neuron. If we need intermediate layer, we can add like `net = network2.Network([50, 30, 5])`. If we want it for minor class classification the last layer should have 55.
7. `net.SGD(training_data, 1, 10, 1.5,lmbda = 0.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)` - This is the core part of the code which actually learns from the input data. See network2.py for its parameters details. You may like / need to modify various parameters of this network to ensure that your ANN learns well.
8. `net.save('networkstore.txt')` saves the network details to `networkstore.txt` file, so that we don't have to repeat the whole training again and again.
9. Reloading the network instructions forthcoming. Code not yet ready.


