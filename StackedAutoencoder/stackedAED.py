# Code written by : Zhecan Wang
# 2016.11.20

import struct
import array
import numpy
import math
import time
import scipy.io
import scipy.optimize
import pickle
import math
import datetime
import os


###########################################################################################
""" Returns elementwise sigmoid output of input array """

def sigmoid(x):

    return (1 / (1 + numpy.exp(-x)))

###########################################################################################
""" The Sparse Autoencoder class """

class SparseAutoencoder(object):

    #######################################################################################
    """ Initialization of Autoencoder object """

    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
    
        """ Initialize parameters of the Autoencoder object """
    
        self.visible_size = visible_size    # number of input units
        self.hidden_size = hidden_size      # number of hidden units
        self.rho = rho                      # desired average activation of hidden units
        self.lamda = lamda                  # weight decay parameter
        self.beta = beta                    # weight of sparsity penalty term
        
        """ Set limits for accessing 'theta' values """
        
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size
        
        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """
        
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        
        rand = numpy.random.RandomState(int(time.time()))
        
        W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        
        """ Bias values are initialized to zero """
        
        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        """ Create 'theta' by unrolling W1, W2, b1, b2 """

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    #######################################################################################
    """ Returns gradient of 'theta' using Backpropagation algorithm """
        
    def sparseAutoencoderCost(self, theta, input):
        
        """ Extract weights and biases from 'theta' input """
        
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        
        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """
        
        hidden_layer = sigmoid(numpy.dot(W1, input) + b1)
        output_layer = sigmoid(numpy.dot(W2, hidden_layer) + b2)
        
        """ Estimate the average activation value of the hidden layers """
        
        rho_cap = numpy.sum(hidden_layer, axis = 1) / input.shape[1]
        
        """ Compute intermediate difference values using Backpropagation algorithm """
        
        diff = output_layer - input
        
        sum_of_squares_error = numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]

        weight_decay         = self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)))

        KL_divergence        = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                                    (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))

        cost                 = sum_of_squares_error + weight_decay 
        # + KL_divergence
        
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        
        del_out = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)), 
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))
        
        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """
            
        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis = 1)
        b2_grad = numpy.sum(del_out, axis = 1)
            
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        
        """ Transform numpy matrices into arrays """
        
        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)
        
        """ Unroll the gradient values and return as 'theta' gradient """
        
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))
                                        
        return [cost, theta_grad]

###########################################################################################
""" The Softmax Regression class """

class SoftmaxRegression(object):

    #######################################################################################
    """ Initialization of Regressor object """

    def __init__(self, input_size, num_classes, lamda):
    
        """ Initialize parameters of the Regressor object """
    
        self.input_size  = input_size  # input vector size
        self.num_classes = num_classes # number of classes
        self.lamda       = lamda       # weight decay parameter
        
        """ Randomly initialize the class weights """
        
        rand = numpy.random.RandomState(int(time.time()))
        
        self.theta = 0.005 * numpy.asarray(rand.normal(size = (num_classes*input_size, 1)))
        
    #######################################################################################
    """ Returns the cost and gradient of 'theta' at a particular 'theta' """
        
    def softmaxCost(self, theta, input, labels):
    
        """ Compute the groundtruth matrix """
        
        # print "labels.shape"
        # print labels.shape


        # ground_truth = getGroundTruth(labels)
        ground_truth = labels
        ground_truth = numpy.transpose(ground_truth)

        """ Reshape 'theta' for ease of computation """
        
        theta = theta.reshape(self.num_classes, self.input_size)
        
        """ Compute the class probabilities for each example """
        
        # theta_x       = numpy.dot(theta, input)
        # hypothesis    = numpy.exp(theta_x)  
        # probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)

        probabilities_revised = numpy.dot(theta, input)
        
        # print "probability_revised.shape"
        # print probabilities_revised.shape

        """ Compute the traditional cost term """
        # print "probabilities.shape"
        # print probabilities.shape
        # print "ground_truth.shape"
        # print ground_truth.shape


        # cost_examples    = numpy.multiply(ground_truth, numpy.log(probabilities))
        # traditional_cost = -(numpy.sum(cost_examples) / input.shape[1])
        
        traditional_cost_revised = numpy.sum(numpy.power(numpy.absolute(probabilities_revised - ground_truth), 2))

        # print "traditional_cost"
        # print type(traditional_cost)
        # print traditional_cost.shape
        # print "traditional_cost_revised"
        # print traditional_cost_revised.shape

        """ Compute the weight decay term """
        
        theta_squared = numpy.multiply(theta, theta)
        weight_decay  = self.lamda * numpy.sum(theta_squared)
        
        """ Add both terms to get the cost """
        
        cost = traditional_cost_revised + weight_decay
        
        """ Compute and unroll 'theta' gradient """
        
        theta_grad = -numpy.dot(ground_truth - probabilities_revised, numpy.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = numpy.array(theta_grad)
        theta_grad = theta_grad.flatten()
        
        return [cost, theta_grad]

    
###########################################################################################
""" Returns the hidden layer activations of the Autoencoder """

def feedForwardAutoencoder(theta, hidden_size, visible_size, input):

    """ Define limits to access useful data """

    limit0 = 0
    limit1 = hidden_size * visible_size
    limit2 = 2 * hidden_size * visible_size
    limit3 = 2 * hidden_size * visible_size + hidden_size
    
    """ Access W1 and b1 from 'theta' """
    
    W1 = theta[limit0 : limit1].reshape(hidden_size, visible_size)
    b1 = theta[limit2 : limit3].reshape(hidden_size, 1)
    
    """ Compute the hidden layer activations """
    
    hidden_layer = 1 / (1 + numpy.exp(-(numpy.dot(W1, input) + b1)))
    
    return hidden_layer


def getOutput(theta, hidden_size, visible_size, input):

    """ Define limits to access useful data """

    limit0 = 0
    limit1 = hidden_size * visible_size
    limit2 = 2 * hidden_size * visible_size
    limit3 = 2 * hidden_size * visible_size + hidden_size
    limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size

    """ Access W1 and b1 from 'theta' """
    W1 = theta[limit0 : limit1].reshape(hidden_size, visible_size)
    W2 = theta[limit1 : limit2].reshape(visible_size, hidden_size)
    b1 = theta[limit2 : limit3].reshape(hidden_size, 1)
    b2 = theta[limit3 : limit4].reshape(visible_size, 1)    
    
    """ Compute the hidden layer activations """
    
    hidden_layer = 1 / (1 + numpy.exp(-(numpy.dot(W1, input) + b1)))

    output_layer = sigmoid(numpy.dot(W2, hidden_layer) + b2)
    
    return output_layer


###########################################################################################
""" Returns a row of Stacked Autoencoder parameters """

def stack2Params(stack):

    """ Initialize an empty list of parameters """

    params = []
    num_layers = len(stack) / 2
    
    """ For each layer in the neural network, append the corresponding parameters """
    
    for i in range(num_layers):
    
        params = numpy.concatenate((params, numpy.array(stack[i, "W"]).flatten()))
        params = numpy.concatenate((params, numpy.array(stack[i, "b"]).flatten()))
        
    return params

###########################################################################################
""" Returns a stack of Stacked Autoencoder parameters """

def params2Stack(params, net_config):

    """ Initialize an empty stack """

    stack  = {}
    limit0 = 0
    
    for i in range(len(net_config)-2):
    
        """ Calculate limits of layer parameters, using neural network config """
    
        limit1 = limit0 + net_config[i] * net_config[i+1]
        limit2 = limit1 + net_config[i+1]
        
        """ Extract layer parameters, and store in the stack """
        
        stack[i, "W"] = params[limit0 : limit1].reshape(net_config[i+1], net_config[i])
        stack[i, "b"] = params[limit1 : limit2].reshape(net_config[i+1], 1)
        
        limit0 = limit2
        
    return stack
    
###########################################################################################
""" Function for finetuning the Stacked Autoencoder """

def stackedAutoencoderCost(theta, net_config, lamda, data, labels):

    """ Calculate limits for Softmax parameters """

    input_size   = net_config[-2]
    num_classes  = net_config[-1]
    
    limit0 = 0
    limit1 = num_classes * input_size
    
    """ Extract Softmax and layer parameters """
    
    softmax_theta = theta[limit0 : limit1].reshape(num_classes, input_size)
    stack         = params2Stack(theta[limit1 :], net_config)
    
    num_layers = len(stack) / 2
    
    """ Calculate activations for every layer """
    
    activation    = {}
    activation[0] = data
    
    for i in range(num_layers):
    
        activation[i+1] = sigmoid(numpy.dot(stack[i, "W"], activation[i]) + stack[i, "b"])
        
    """ Compute the groundtruth matrix """
    
    # ground_truth = getGroundTruth(labels)
    ground_truth = labels
    ground_truth = numpy.transpose(ground_truth)

    """ Compute the class probabilities for each example """
    
    # theta_x       = numpy.dot(softmax_theta, activation[num_layers])
    # hypothesis    = numpy.exp(theta_x)      
    # probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
    probabilities_revised = numpy.dot(softmax_theta, activation[num_layers])
    
    """ Compute the traditional cost term """
    
    # cost_examples    = numpy.multiply(ground_truth, numpy.log(probabilities))
    # traditional_cost = -(numpy.sum(cost_examples) / data.shape[1])
    traditional_cost_revised = numpy.sum(numpy.power(numpy.absolute(probabilities_revised - ground_truth), 2))
    
    """ Compute the weight decay term """
    
    theta_squared = numpy.multiply(softmax_theta, softmax_theta)
    weight_decay  = lamda * numpy.sum(theta_squared)
    
    """ Add both terms to get the cost """
    
    cost = traditional_cost_revised + weight_decay
    
    """ Compute Softmax 'theta' gradient """
    
    softmax_theta_grad = -numpy.dot(ground_truth - probabilities_revised, numpy.transpose(activation[num_layers]))
    softmax_theta_grad = softmax_theta_grad / data.shape[1] + lamda * softmax_theta
    
    """ Compute intermediate difference values using Backpropagation algorithm """
    
    delta = {}    
    delta[num_layers] = -numpy.multiply(numpy.dot(numpy.transpose(softmax_theta), ground_truth - probabilities_revised),
                                        numpy.multiply(activation[num_layers], 1 - activation[num_layers]))
    for i in range(num_layers-1):
    
        index        = num_layers - i - 1
        delta[index] = numpy.multiply(numpy.dot(numpy.transpose(stack[index, "W"]), delta[index+1]),
                                       numpy.multiply(activation[index], 1 - activation[index]))
                                       
    """ Compute the partial derivatives, with respect to the layer parameters """
                                       
    stack_grad = {}
    
    for i in range(num_layers):
    
        index = num_layers - i - 1
        stack_grad[index, "W"] = numpy.dot(delta[index+1], numpy.transpose(activation[index])) / data.shape[1]
        stack_grad[index, "b"] = numpy.sum(delta[index+1], axis = 1) / data.shape[1]
    
    """ Concatenate the gradient values and return as 'theta' gradient """
        
    params_grad = stack2Params(stack_grad)
    theta_grad  = numpy.concatenate((numpy.array(softmax_theta_grad).flatten(),
                                     numpy.array(params_grad).flatten()))
    
    return [cost, theta_grad]
    
###########################################################################################
""" Returns predictions using the trained Stacked Autoencoder model """

def stackedAutoencoderPredict(theta, net_config, data):

    """ Calculate limits for Softmax parameters """

    input_size   = net_config[-2]
    num_classes  = net_config[-1]
    
    limit0 = 0
    limit1 = num_classes * input_size
    
    """ Extract Softmax and layer parameters """
    
    softmax_theta = theta[limit0 : limit1].reshape(num_classes, input_size)
    stack         = params2Stack(theta[limit1 :], net_config)
    
    num_layers = len(stack) / 2
    
    """ Calculate the activations of the final layer """
    
    activation = data
    
    for i in range(num_layers):
    
        activation = sigmoid(numpy.dot(stack[i, "W"], activation) + stack[i, "b"])
        
    """ Compute the class probabilities for each example """
        
    # theta_x       = numpy.dot(softmax_theta, activation)
    # hypothesis    = numpy.exp(theta_x)      
    # probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)

    probabilities_revised = numpy.dot(softmax_theta, activation)
    
    """ Give the predictions based on probability values """
    
    # predictions       = numpy.zeros((data.shape[1], 136))
    # predictions[:, 0] = numpy.argmax(probabilities_revised, axis = 0)
    predictions = probabilities_revised

    return predictions

###########################################################################################
""" Loads data, trains the Stacked Autoencoder model and predicts classes for test data """

def executeStackedAutoencoder():
    
    """ Define the parameters of the first Autoencoder """
    
    visible_size   = 2500    # size of input vector
    hidden_size1   = 1600    # size of hidden layer vector of first autoencoder
    hidden_size2   = 900    # size of hidden layer vector of second autoencoder
    hidden_size3   = 400    # size of hidden layer vector of second autoencoder

    rho            = 0.1    # desired average activation of hidden units
    lamda          = 0.001  # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    max_iterations = 600    # number of optimization iterations
    num_classes    = 136     # number of classes
    
    """ Load MNIST images for training and testing """
    

    # train_data = pickle.load( open("./data/xTrainFlattenSpec.p", "rb" ) )[:30]
    # train_labels = pickle.load( open("./data/yTrainSpec.p", "rb" ) )[:30]
    train_data = pickle.load( open("./data/xTrainFlattenSpec.p", "rb" ) )
    train_labels = pickle.load( open("./data/yTrainSpec.p", "rb" ) )
    (num, d) = train_data.shape
    train_data = train_data.reshape((d, num))

    print train_data.shape
    print train_labels.shape


    """ Initialize the first Autoencoder with the above parameters """
    
    encoder1 = SparseAutoencoder(visible_size, hidden_size1, rho, lamda, beta)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """

    opt_solution   = scipy.optimize.minimize(encoder1.sparseAutoencoderCost, encoder1.theta, 
                                             args = (train_data,), method = 'L-BFGS-B', 
                                             jac = True, options = {'maxiter': max_iterations, "disp": True})

    # , "disp": True
    sae1_opt_theta = opt_solution.x
    
    """ Get the features corresponding to first Autoencoder """
    
    sae1_features = feedForwardAutoencoder(sae1_opt_theta, hidden_size1, visible_size, train_data)

    print "######################################## finish the first AED ########################################"   

    # store the recovered face from first layer autoencoder
    output = getOutput(sae1_opt_theta, hidden_size1, visible_size, train_data)
    (num, d) = output.shape
    output = output.reshape((d, num))
    if not os.path.isfile("./data/output/firstAEDoutput" + str(datetime.datetime.now()) + ".p"):
        print "########################################   save the recovered photo ########################################"
        pickle.dump( output, open( "./data/output/firstAEDoutput" + str(datetime.datetime.now()) + ".p", "wb" ) )  
     # 

    """ Initialize the second Autoencoder with the above parameters """
    
    encoder2 = SparseAutoencoder(hidden_size1, hidden_size2, rho, lamda, beta)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    
    opt_solution   = scipy.optimize.minimize(encoder2.sparseAutoencoderCost, encoder2.theta, 
                                             args = (sae1_features,), method = 'L-BFGS-B', 
                                             jac = True, options = {'maxiter': max_iterations, "disp": True})
    sae2_opt_theta = opt_solution.x
    
    """ Get the features corresponding to second Autoencoder """
    
    sae2_features = feedForwardAutoencoder(sae2_opt_theta, hidden_size2, hidden_size1, sae1_features)

    print "######################################## finish the second AED ########################################"   
    

    """ Initialize the third Autoencoder with the above parameters """
    
    encoder3 = SparseAutoencoder(hidden_size2, hidden_size3, rho, lamda, beta)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    
    opt_solution   = scipy.optimize.minimize(encoder3.sparseAutoencoderCost, encoder3.theta, 
                                             args = (sae2_features,), method = 'L-BFGS-B', 
                                             jac = True, options = {'maxiter': max_iterations, "disp": True})
    sae3_opt_theta = opt_solution.x
    
    """ Get the features corresponding to second Autoencoder """
    
    sae3_features = feedForwardAutoencoder(sae3_opt_theta, hidden_size3, hidden_size2, sae2_features)

    print "######################################## finish the third AED ########################################"   
    
    """ Initialize Softmax Regressor with the above parameters """
    
    regressor = SoftmaxRegression(hidden_size3, num_classes, lamda)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    
    opt_solution      = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta, 
                                                args = (sae3_features, train_labels,), method = 'L-BFGS-B', 
                                                jac = True, options = {'maxiter': max_iterations, "disp": True})
    softmax_opt_theta = opt_solution.x
    
    """ Create a stack of the Stacked Autoencoder parameters """
    
    stack = {}
    stack[0, "W"] = sae1_opt_theta[encoder1.limit0 : encoder1.limit1].reshape(hidden_size1, visible_size)
    stack[1, "W"] = sae2_opt_theta[encoder2.limit0 : encoder2.limit1].reshape(hidden_size2, hidden_size1)
    stack[2, "W"] = sae3_opt_theta[encoder3.limit0 : encoder3.limit1].reshape(hidden_size3, hidden_size2)

    stack[0, "b"] = sae1_opt_theta[encoder1.limit2 : encoder1.limit3].reshape(hidden_size1, 1)
    stack[1, "b"] = sae2_opt_theta[encoder2.limit2 : encoder2.limit3].reshape(hidden_size2, 1)
    stack[2, "b"] = sae3_opt_theta[encoder3.limit2 : encoder3.limit3].reshape(hidden_size3, 1)

    
    """ Create a vector of the Stacked Autoencoder parameters for optimization """
    
    stack_params     = stack2Params(stack)
    stacked_ae_theta = numpy.concatenate((softmax_opt_theta.flatten(), stack_params.flatten()))
    
    """ Create a neural network configuration, with number of units in each layer """
    
    net_config = [visible_size, hidden_size1, hidden_size2, hidden_size3, num_classes]
    
    """ Load test images and labels """


    # test_data = pickle.load( open( "./data/xTestFlattenSpec.p", "rb" ) )[:1000]
    # test_labels = pickle.load( open( "./data/yTestSpec.p", "rb" ) )[:1000]
    test_data = pickle.load( open( "./data/xTestFlattenSpec.p", "rb" ) )
    test_labels = pickle.load( open( "./data/yTestSpec.p", "rb" ) )
    (num, d) = test_data.shape
    test_data = test_data.reshape((d, num))
    
    print test_data.shape
    print test_labels.shape

########################################################################################
    """ Get predictions after greedy training """

    unSupervisedTrainPred = stackedAutoencoderPredict(stacked_ae_theta, net_config, train_data)
    unSupervisedTrainPred = numpy.transpose(unSupervisedTrainPred)

    unSupervisedTestPred = stackedAutoencoderPredict(stacked_ae_theta, net_config, test_data)
    unSupervisedTestPred = numpy.transpose(unSupervisedTestPred)

    pickle.dump( unSupervisedTrainPred, open( "./data/output/unSupervisedTrainPred" + str(datetime.datetime.now()) + ".p", "wb" ) )    
    pickle.dump( unSupervisedTestPred, open( "./data/output/unSupervisedTestPred" + str(datetime.datetime.now()) + ".p", "wb" ) )
    
    """ Print accuracy of the trained model """
    unSupervisedTrainAccuracy = getAcurracy(train_labels, unSupervisedTrainPred)
    print """Train Data Accuracy after greedy training :""", unSupervisedTrainAccuracy

    unSupervisedTestAccuracy = getAcurracy(test_labels, unSupervisedTestPred)
    print """Test Data Accuracy after greedy training :""", unSupervisedTestAccuracy


########################################################################################    
    """ Finetune the greedily trained model """
    
    opt_solution = scipy.optimize.minimize(stackedAutoencoderCost, stacked_ae_theta, 
                                           args = (net_config, lamda, train_data, train_labels,),
                                           method = 'L-BFGS-B', jac = True, options = {'maxiter': max_iterations, "disp": True})
    stacked_ae_opt_theta = opt_solution.x
    
    """ Get predictions after finetuning """
    
    supervisedTrainPred = stackedAutoencoderPredict(stacked_ae_opt_theta, net_config, train_data)
    supervisedTrainPred = numpy.transpose(supervisedTrainPred)

    supervisedTestPred = stackedAutoencoderPredict(stacked_ae_opt_theta, net_config, test_data)
    supervisedTestPred = numpy.transpose(supervisedTestPred)
    
    pickle.dump( supervisedTrainPred, open( "./data/output/supervisedTrainPred" + str(datetime.datetime.now()) + ".p", "wb" ) )    
    pickle.dump( supervisedTestPred, open( "./data/output/supervisedTestPred" + str(datetime.datetime.now()) + ".p", "wb" ) )
    

    """ Print accuracy of the trained model """
    supervisedTrainAccuracy = getAcurracy(train_labels, supervisedTrainPred)
    print """Train Data Accuracy after finetuning :""", supervisedTrainAccuracy

    supervisedTestAccuracy = getAcurracy(test_labels, supervisedTestPred)
    print """Test Data Accuracy after finetuning :""", supervisedTestAccuracy
 

def getAcurracy(test_labels, predictions):
    newPredictions = []
    newLabels = []
    error = []
    (length, _) = predictions.shape
    for index in range(length):
        pred = predictions[index]
        x, y =parseLocation(pred)
        predPts = zip(x, y)
        newPredictions.append(predPts)
        label = test_labels[index]
        x, y =parseLocation(label)
        labelPts = zip(x, y)   
        newLabels.append(labelPts)

        interocular_distance = math.sqrt((labelPts[37][0] - labelPts[46][0])**2 + (labelPts[37][1] - labelPts[46][1])**2)


        cum = 0
        for i in range(68):
            cum = cum + math.sqrt((predPts[i][0] - labelPts[i][0])**2 + (predPts[i][1] - labelPts[i][1])**2);

        error.append(cum/(68*interocular_distance))

    error = numpy.asarray(error)
    correct = numpy.mean(error)

    # error = numpy.mean(numpy.absolute(test_labels - predictions))
    # correct = 1.0 - error
    
    # correct = test_labels[:, 0] == predictions[:, 0]

    return correct    
 

def parseLocation(array):
    x = []
    y = []
    for i in range(0, len(array), 2):
        x.append(array[i])
        y.append(array[i + 1])
    return x, y

if __name__ == '__main__':
    executeStackedAutoencoder()
