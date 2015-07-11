Network to identify the major classes of samAsas

cd codes/gittest/anndev/src
python -i
import trialcode
training_data, validation_data, test_data = trialcode.load_data_wrapper_major(trialcode.inputwords,trialcode.majorwords)
import network2
net = network2.Network([50, 5])
net.SGD(training_data, 1, 10, 1.5,lmbda = 0.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)
net.save('networkstore.txt')


----------------------------
Network to identify the minor classes of samAsas (yet to be done)

cd codes/gittest/anndev/src
python -i
import trialcode
training_data, validation_data, test_data = trialcode.load_data_wrapper(trialcode.inputwords,trialcode.outputwords)
import network2
net = network2.Network([50, 50, 55])
net.SGD(training_data, 30, 10, 1.5,lmbda = 0.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True, items=1)
net.save('networkstore.txt')


----------------------------

To Load from Stored network

cd codes/gittest/anndev/src
python -i
import trialcode
training_data, validation_data, test_data = trialcode.load_data_wrapper(trialcode.inputwords,trialcode.outputwords)
import network2
net = network2.load('networkstore.txt')
net.SGD(training_data, 1, 10, 0.75,lmbda = 0.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

