Network after ASCII generalisation

cd codes/SamaasaClassification
python -i
import preparation as p
training_data, test_data = p.load_data_wrapper(p.inputwords, p.outputwords, p.output_neuron, p.mode, p.input_neuron)
import network2
net = network2.Network([p.input_neuron, p.intermediate_neuron, p.output_neuron])
net.SGD(training_data, p.epochs, p.mini_batch_size, p.eta, p.all_class_types, p.mode, p.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=p.howmany)
import OutputDataAnalyser as n
n.outputdataanalysis("outputdata.txt", p.all_class_types)
----------------------------

To load and train again from stored network

cd codes/SamaasaClassification
python -i
import preparation as p
training_data, test_data = p.load_data_wrapper(p.inputwords,p.outputwords, p.output_neuron, p.mode, p.input_neuron)
import network2
net = network2.load('networkstore.txt')
net.SGD(training_data, p.epochs, p.mini_batch_size, p.eta, p.all_class_types, p.mode, p.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=p.howmany)

----------------------------

To load from stored network and test accuracy
# Before loading, make necessary changes in preparation.py to change inputwords and outputwords etc

cd codes/SamaasaClassification
python -i
import preparation as p
import network2
net = network2.load('networkstore.txt')
training_data, test_data = p.load_data_wrapper(p.inputwords, p.outputwords, p.output_neuron, p.mode, p.input_neuron)
net.testing(test_data, p.howmany)

