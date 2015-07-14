Network to identify the major classes of samAsas

cd codes/SamaasaClassification
python -i
import preparation
training_data, test_data = preparation.load_data_wrapper(preparation.inputwords,preparation.outputwords)
import network2
net = network2.Network([50, 150, 55])
net.SGD(training_data, 1, 10, 1.5,lmbda = 0.0, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)
net.save('networkstore.txt')


----------------------------
Trial Network after ASCII generalisation

cd codes/SamaasaClassification
python -i
import preparation
training_data, test_data = preparation.load_data_wrapper(preparation.inputwords, preparation.outputwords, preparation.output_neuron, preparation.mode)
import network2
net = network2.Network([preparation.input_neuron, preparation.intermediate_neuron, preparation.output_neuron])
net.SGD(training_data, preparation.epochs, preparation.mini_batch_size, preparation.eta, preparation.all_class_types, preparation.mode, preparation.lmbda, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=preparation.howmany)
net.save('networkstore.txt')


----------------------------
Network to identify the minor classes of samAsas (yet to be done)

cd codes/SamaasaClassification
python -i
import preparation
training_data, test_data = preparation.load_data_wrapper(preparation.inputwords,preparation.outputwords)
import network2
net = network2.Network([50, 50, 55])
net.SGD(training_data, 30, 10, 1.5,lmbda = 0.0, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)
net.save('networkstore.txt')


----------------------------

To Load from Stored network

cd codes/SamaasaClassification
python -i
import preparation
training_data, test_data = preparation.load_data_wrapper(preparation.inputwords,preparation.outputwords)
import network2
net = network2.load('networkstore.txt')
net.SGD(training_data, 1, 10, 0.75,lmbda = 0.0, evaluation_data=test_ldata, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

