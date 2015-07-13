Network to identify the major classes of samAsas

cd codes/SamaasaClassification
python -i
import trialcode
training_data, test_data = trialcode.load_data_wrapper_major(trialcode.inputwords,trialcode.majorwords)
import network2
net = network2.Network([50, 50, 5])
net.SGD(training_data, 30, 10, 1.5,lmbda = 0.0, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)
net.save('networkstore.txt')


----------------------------
Network to identify the minor classes of samAsas (yet to be done)

cd codes/SamaasaClassification
python -i
import trialcode
training_data, test_data = trialcode.load_data_wrapper(trialcode.inputwords,trialcode.outputwords)
import network2
net = network2.Network([50, 50, 55])
net.SGD(training_data, 30, 10, 1.5,lmbda = 0.0, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)
net.save('networkstore.txt')


----------------------------

To Load from Stored network

cd codes/SamaasaClassification
python -i
import trialcode
training_data, test_data = trialcode.load_data_wrapper(trialcode.inputwords,trialcode.outputwords)
import network2
net = network2.load('networkstore.txt')
net.SGD(training_data, 1, 10, 0.75,lmbda = 0.0, evaluation_data=test_ldata, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

