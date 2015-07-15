Network after ASCII generalisation

cd codes/SamaasaClassification
python -i
import preparation as p
training_data, test_data = p.load_data_wrapper(p.inputwords, p.outputwords, p.output_neuron, p.mode)
import network2
net = network2.Network([p.input_neuron, p.intermediate_neuron, p.output_neuron])
net.SGD(training_data, p.epochs, p.mini_batch_size, p.eta, p.all_class_types, p.mode, p.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=p.howmany)
net.save('networkstore.txt')

----------------------------

To load from stored network

cd codes/SamaasaClassification
python -i
import preparation
training_data, test_data = p.load_data_wrapper(p.inputwords,p.outputwords)
import network2
net = network2.load('networkstore.txt')
net.SGD(training_data, p.epochs, p.mini_batch_size, p.eta, p.all_class_types, p.mode, p.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=p.howmany)


