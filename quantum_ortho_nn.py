#### Libraries
# Standard library
import json
import random
import sys
import time

# Third-party libraries
import numpy as np

####useful to convert gate number into pair of indices ### NOT NEEDED
def convert_gate_number_to_pair_of_indices(gate_num,n):
    """n=total number of gates. gate_num is between 1 and n
    returns : (j,k) where j is the qubit wire n°, and k is the position of the gate on this wire
    note : j & k are counted from 0"""
    first_gates_num = [2 + ((k+1)**2)/2 + - 3*(k+1)/2 for k in range(1,n+1)] # global: the number of the first gate of each wire
    diag_of_gate = np.where(np.array(first_gates_num)>gate_num)[0][0]-1 #find the diagonal (bot-left to top-rigt) to which the gate belongs 
    position_of_gate = int(gate_num - first_gates_num[diag_of_gate]) #position (n°) of the gate on the diagonal. 0 is first = bottom gate
    #print('QUANTUM...')('\ngate_num :',gate_num,' | diag_of_gate :',diag_of_gate,' | position_of_gate :',position_of_gate)
    return (diag_of_gate-position_of_gate, position_of_gate) #(j,k)


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * (1.0/(1.0+np.exp(-z)))*(1-1.0/(1.0+np.exp(-z)))


class CrossEntropyCost(object):

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
    

#### Define the sigmoid and Relu activation functions
class SigmoidActivation(object):

    @staticmethod
    def fn(z):
        """The sigmoid function."""
        z = z.astype(float)
        for i in z:
            if i[0]>500:
                i[0]=500
            if i[0]<-500:
                i[0]=-500
        return 1.0/(1.0+np.exp(-z))

    def prime(z):
        """Derivative of the sigmoid function."""
        z = z.astype(float)
        for i in z:
            if i[0]>300:
                i[0]=300
            if i[0]<-300:
                i[0]=-300
        return np.exp(-z)/(1.0+np.exp(-z))**2

class ReLuActivation(object):

    @staticmethod
    def fn(z):
        """The ReLu function."""
        return np.maximum(0.0,z)

    def prime(z):
        """Derivative of the ReLu function."""
        for i in range(z.shape[0]):
            if z[i]<= 0.0:
                z[i]=0.0
            else:
                z[i]=1.0
        return z

    

#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, activation=SigmoidActivation):
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
        self.default_weight_initializer() #initialize angles (and generate weights fromt them) and biases
        self.cost=cost
        self.activation=activation

        self.evaluation_cost= [] 
        self.evaluation_accuracy = []
        self.training_cost= [] 
        self.training_accuracy = []

    def default_weight_initializer(self):
        """Initialize randomly the anlges of the quantum circuit, for each layer. 
        Use them to generate the weight for each layer of the network. 
        
        Initialize the biases as well using a Gaussian distribution with mean 0 and standard
        deviation 1. Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        [new] Note : for x, y in zip(self.sizes[:-1], self.sizes[1:]) is a loop over 
        each layer where x is the input size and y is the output size.
        """
        

        #### 1 - angles
    #    print("Generate angles :")
        # /!!!\ here we generate too many angles (as if it was a square case) but it's helpful. Some of them are left unused...
        angle_range = np.arange(0,2*np.pi,np.pi*1e-6) #we need discretized angle space to avoic numpy numerical errors for tiny values : create a range of possible angles a priori, with a 10^{-6} step        
#        self.angles = [np.random.choice(angle_range, int(x*(x-1)/2)) for x in self.sizes[:-1]] #### note we use x(x-1)/2 and not y*(y-1)/2 + y*(x-y) 
    #    print("angles shape :",[self.angles[i].shape for i in range(len(self.angles))])
        # for i in range(len(self.angles)):
        #     print("\nangles",i,"\n",self.angles[i])
    #    print("Angles represent",np.sum(self.angles),"tunable parameters")

        self.angles = [np.random.random_integers(0,3,int(x*(x-1)/2))*np.pi/2+np.pi/4+np.random.normal(0,0.01,int(x*(x-1)/2)) for x in self.sizes[:-1]]
    
    
        ### 2 - weights
        #SERIAL MATRIX MULTIPLICATION VERSION
        #Generate all necessary matrices and matrix multiplications
    #    print("\nGenerate_weights_from_angles_matrix_mult:")
        results = [self.Generate_weights_from_angles_matrix_mult(angles_layer_list,input_size,output_size) 
                                for angles_layer_list,input_size,output_size in zip(self.angles, self.sizes[:-1], self.sizes[1:])]
        
        self.angles_matrix_list = np.array([results[i][0] for i in range(self.num_layers-1)], dtype=object)
        self.weights = np.array([results[i][1] for i in range(self.num_layers-1)], dtype=object)
        self.angles = np.array([results[i][2] for i in range(self.num_layers-1)], dtype=object)

        ### 3 - biases
    #    print("\nGenerate biases :")
#         self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]



###################################################################################################
     # FUNCTIONS TO GENERATE THE WEIGHTS FROM THE ANGLES
    #### SERIAL MATRIX MULTIPLICATIONS Creation


    def Generate_weights_from_angles_matrix_mult(self, angles_layer_list, input_size, output_size):
        """Here we generate few things :
        - we store each "mini layer", which is the binary state after applying each timestep (an vertical step)
        - at the end we need the weight matrix corresponding to the full layer from the angles, 
        using many intermediate matrix multiplications corresponding to each
        gate with its angle. Beware of the order when multiplying. """

        ### 1 - reshape the layer's angles as a matrix
        ### 2 - fast creation of W, the full weight matrix

#         max_gate_index = int(input_size*(input_size-1)/2) # int(x*(x-1)/2) = total number of gates
        
        all_angles = np.zeros(len(angles_layer_list))
        W = np.eye(input_size) #intialized as identity at first
        
        angles_layer_matrix = [[0 for j in range(i)] for i in range(input_size-1,0,-1)]
        for i in range(input_size-1):
            for j in range(min(i+1,output_size)):       
                angle_num = int(i*(i+1)/2 + j)
                angles_layer_matrix[i-j][j] = angles_layer_list[angle_num]
                W = self.gate(i-j, angles_layer_matrix[i-j][j], W) #add a new gate #O(constant)
                all_angles[angle_num] = angles_layer_list[angle_num]

        W = W[-output_size:] #need to crop for rectangular matrices. It should be output_size x input_size !

        return [angles_layer_matrix, W, all_angles]


    def gate(self, i, theta, vec): 
        """Apply the i^th gate on vector "vec" with angle theta. Returns a vector
        note : for Wbar, just apply gate(i,theta+np.pi/2,vec)!"""
        time_0 = time.time()
        if theta == 0:
            return vec
        vec_i_new = vec[i]*np.cos(theta) + vec[i+1]*np.sin(theta)
        vec_i_plus_1_new = - vec[i]*np.sin(theta) + vec[i+1]*np.cos(theta)
        vec[i:i+2] = [vec_i_new, vec_i_plus_1_new]
        return vec


##########################################################################################

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation.fn(np.dot(w, a)+b) #a = sigmoid(z)
        return a
       

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_test_cost=False,
            monitor_test_accuracy=True,
            monitor_training_cost=False,
            monitor_training_accuracy=True,
            print_during_training = False,
            early_stopping_n = 0):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        test data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the test data, the
        accuracies on the test data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        test data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """

        # early stopping functionality:
        best_accuracy=1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0

        test_cost, test_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        #print('**Before Training :')
        if monitor_training_cost:
            cost = self.total_cost(training_data, lmbda)
            training_cost.append(cost)
            if print_during_training:
                print("\tCost on training data: {}".format(cost))
        if monitor_training_accuracy:
            accuracy = self.accuracy(training_data, convert=True)
            training_accuracy.append(accuracy)
            if print_during_training:
                print("\tAccuracy on training data: {} / {}".format(accuracy, n))
        if monitor_test_cost:
            cost = self.total_cost(evaluation_data, lmbda, convert=True)
            test_cost.append(cost)
            if print_during_training:
                print("\tCost on test data: {}".format(cost))
        if monitor_test_accuracy:
            accuracy = self.accuracy(evaluation_data, convert=True)
            test_accuracy.append(accuracy)
            if print_during_training:
                print("\tAccuracy on test data: {} / {}".format(self.accuracy(evaluation_data, convert=True), n_data))
        #print('---------\n')


        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            ##### HERE WE UPDATE THE PARAMETERS
            count = 1
            for mini_batch in mini_batches:
                #print('minibatch {}/{}'.format(count,len(mini_batches)), end='\r')
                count+=1
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                if print_during_training:
                    print("\tCost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                if print_during_training:
                    print("\tAccuracy on training data: {} / {}".format(accuracy, n))
            if monitor_test_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                test_cost.append(cost)
                if print_during_training:
                    print("\tCost on test data: {}".format(cost))
            if monitor_test_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                test_accuracy.append(accuracy)
                if print_during_training:
                    print("\tAccuracy on test data: {} / {}".format(self.accuracy(evaluation_data, convert=True), n_data))

        self.evaluation_cost = test_cost
        self.evaluation_accuracy = test_accuracy
        self.training_cost = training_cost
        self.training_accuracy = training_accuracy

        return test_cost, test_accuracy, training_cost, training_accuracy

#####################################################################################
## NEW BACKPROP AND UPDATE OF THE PARAMETERS 

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """

        #define two list of list, which will contain the gradient of all parameters (angles or bias) at each layer
        #we will add up new gradients value by doing backprop on one sample at a time 
        nabla_angles = [np.zeros(a.shape) for a in self.angles] #NEW! replace angles
        nabla_bias = [np.zeros(b.shape) for b in self.biases] 
        count = 0
        for x, y in mini_batch:
            """x is one sample of the data itself = one input vector, and y is the desired/true ouput vector
            we do backprop for each sample, add them up, and after this loop we will update the parameters with the average gradient"""
            delta_nabla_bias, delta_nabla_angles = self.backprop(x, y) ## HERE WE BACKPROP on one sample
            nabla_angles = [na+np.array(dna).reshape(-1) for na, dna in zip(nabla_angles, delta_nabla_angles)] #add the new gradient value to the previous ones
            nabla_bias = [nb+np.array(dnb) for nb, dnb in zip(nabla_bias, delta_nabla_bias)] #add the new gradient value to the previous ones
        
        #provisoire (stats only)
        self.nabla_angles = nabla_angles

        #update angles!
        self.angles = [(1-eta*(lmbda/n))*a-(eta/len(mini_batch))*na
                        for a, na in zip(self.angles, nabla_angles)]
        #update biases!
#         self.biases = [b-(eta/len(mini_batch))*nb
#                        for b, nb in zip(self.biases, nabla_bias)]

        #now generate again the intermediate matrices, and use them to update the weights as well
        results = [self.Generate_weights_from_angles_matrix_mult(angles_layer_list,input_size,output_size) 
                                for angles_layer_list,input_size,output_size in zip(self.angles, self.sizes[:-1], self.sizes[1:])]
        
        self.angles_matrix_list = np.array([results[i][0] for i in range(self.num_layers-1)], dtype=object)
        self.weights = np.array([results[i][1] for i in range(self.num_layers-1)], dtype=object)

        

    def backprop(self, x, y):
        """ the input x is a data sample, y is it's truth output.
        Return a tuple ``(nabla_b, nabla_angles)`` representing the
        gradient for the cost function C_x. Note that there is not nabla_weights anymore!!!
        ``nabla_b`` and ``nabla_angles`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.angles``.
        """

        #will contain the gradients of all layers' parameters. 
        nabla_bias = [np.zeros(b.shape) for b in self.biases] #it's a list of list
        nabla_angles = [np.zeros(a.shape) for a in self.angles] #it's a list of list

        # FORWARD PASS - Calculate all activations (all layers)

        activation = x #input vector
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer # note: z is also the layer, but just before the sigmoid
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #apply weight matrix and bias
            zs.append(z) #add to the list of z
            activation = self.activation.fn(z) #apply sigmoid
            activations.append(activation) #add to the list of activations


        ###### BACKWARD PASS (all layers)


        ##### LAST LAYER #### 
        # print("\nLAYER",self.num_layers-1," (LAST)", self.sizes[-2],'->',self.sizes[-1])

        #first calculate the error at layer L (the last) (if indexing from 1)
        L = self.num_layers-1 #last layer (indexed from 0)
        Delta_L = (self.cost).delta(zs[-1], activations[-1], y) #calculate the overall resulting error! 

        ### NEW - calculate mini layers !
        ### AT LAST LAYER, perform the evolution at each timestep of the current (big) layer and store the mini layers. They correspond to the right slices of the circuit. in O(?)
        all_mini_layers_LAST = self.calculate_mini_layers(layer_input = activations[L-1],
                                                        input_size = self.sizes[L-1],
                                                        output_size = self.sizes[L],
                                                        angles_layer_matrix = self.angles_matrix_list[L-1])

        #BIAS gradient:
        nabla_bias[-1] = Delta_L         
        
        #ANGLES gradients:
        input_size = self.sizes[L-1]
        output_size = self.sizes[L]
        angles_layer_matrix = self.angles_matrix_list[L-1]

        
        nabla_angles_list_layer = np.zeros(int(input_size*(input_size-1)/2)) #we will put here all gradients (list)
        
        mini_delta = np.zeros(input_size - output_size)
        mini_delta = np.append(mini_delta,Delta_L) #this is the mini delta vector at each timestep, will be updated during the loop, starts as "big" delta because it's the last mini layer = the end of the layer.
        
        angles_layer_matrix = self.angles_matrix_list[-1] 
        for lambda_ in range(2*input_size-4,-1,-1): # Here we loop for every mini layer/timestep LAMBDA from last to first (instead of looping for every angle)

            # 1- Calculate the w_lambda for this lambda (timestep)            
            w_lambda = np.eye(input_size)
            i = lambda_%2 #top gate of the timestep lambda_ (with notation j^th gate of the i^th qubit)
            j = int((lambda_ - i)/2) #top gate of the timestep lambda_
            
            if j>output_size:
                j = output_size
                i = lambda_ - 2*j


########################################  
            while i+j <= input_size-2 and i >= 0 and j >= 0: #Loop over all gates in each timestep: at most O(n/2)
                if angles_layer_matrix[i][j] != 0: #if the gate's angle is not 0
                    #update incrementally the current matrix of the current timestep
                    
                    w_lambda = self.gate(i, angles_layer_matrix[i][j], w_lambda) #O(c)
                #move to the next gate of the current timestep
                i += 2 #two qubits below
                j -= 1 #therefore the gate is now closer to the left
########################################             
            
            
            
            # 2- we can now calculate the gradient for each gate in the timestep
######################################## 
            i = lambda_%2 #top gate of the timestep (with notation j^th gate of the i^th qubit)
            j = int((lambda_ - i)/2) #top gate of the timestep
            
            if j>output_size:
                j = output_size
                i = lambda_ - 2*j
            
            while i+j <= input_size-2 and i >= 0 and j >= 0: #Loop over all gates in each timestep: at most O(n/2)
                angle = angles_layer_matrix[i][j]
                angle_num = int((i+j)*(i+j+1)/2 + j)
                #for this gate n°angle_num, we can now calculate dC/dtheta_angle_num :
                mini_layer_previous = all_mini_layers_LAST[lambda_] #=zeta_lambda
                nabla_angle_i = mini_delta[i]*(- np.sin(angle)*mini_layer_previous[i] + np.cos(angle)*mini_layer_previous[i+1]) + mini_delta[i+1]*(- np.cos(angle)*mini_layer_previous[i] - np.sin(angle)*mini_layer_previous[i+1]) 
                nabla_angles_list_layer[angle_num] = nabla_angle_i #store gradient
 
                #move to the next gate of the current timestep
                i += 2 #two qubits below
                j -= 1 #therefore the gate is now closer to the left
########################################          


            # 3- then calculate the mini_delta_lambda at the timestep, by applying the w_lambda transposed to the mini_delta_lambda+1
            mini_delta = w_lambda.T@mini_delta 
            
        # 4- save this layer's angle gradients   
        nabla_angles[L-1] =  nabla_angles_list_layer #add this (big)layer's angle gradient in the global list

        



        #### PREVIOUS LAYERS #### 

        for l in range(L-1,0,-1): #loop, at layer l between L-2 and 0 (reverse order), because we index from 0, and because the last layer is already done!
            # print("\nLAYER",l,':', self.sizes[l-1],"->",self.sizes[l])            
            # first we need to calculate the error at layer l
            input_size = self.sizes[l-1]
            output_size = self.sizes[l]
            z = zs[l-1]

            sp = self.activation.prime(z).reshape(-1)        

            #pad the sigmoid prime of z to match the input size
            sp_padded = np.zeros(input_size - output_size)
            sp_padded = np.append(sp_padded,sp)
            
            #pad the bias to match the input size
            bias_padded = np.zeros(input_size - output_size)
            bias_padded = np.append(bias_padded,self.biases[l-1])            

            #pad the Delta to match the input size
            Delta = np.zeros(input_size - output_size) #initialize the "virtual" part that should be zeros
            Delta = np.append(Delta,mini_delta) #this is the mini delta vector at each timestep, will be updated during the loop, starts as "big" delta because it's the last mini layer = the end of the layer.

            #new method to compute Delta           
            Delta = Delta * sp_padded #+ bias_padded #THIS SHOULD BE EQUIVALENT

            #BIAS gradient it's exactly Delta, but need to reshape to have a size (...,1)
            nabla_bias[l-1] = Delta.reshape(len(Delta),1)[-output_size:]

            ### NEW - calculate ALL mini layers (for all timesteps / lambdas)!
            ### AT THIS LAYER, perform the evolution at each timestep of the current (big) layer and store the mini layers. They correspond to the right slices of the circuit. in O(?)
            ### REMARK : we could change this function to return only a pair of values by existing gate in the timestep, because we don't need the others (save memory!!)
            all_mini_layers = self.calculate_mini_layers(layer_input = activations[l-1], 
                                                        input_size = self.sizes[l-1], 
                                                        output_size = self.sizes[l], 
                                                        angles_layer_matrix = self.angles_matrix_list[l-1]) 
            

            nabla_angles_list_layer = np.zeros(int(input_size*(input_size-1)/2)) #we will put here all gradients of the current Layer (list)
            mini_delta = Delta
            angles_layer_matrix = self.angles_matrix_list[l-1] 
            for lambda_ in range(2*input_size-4,-1,-1): # Here we BACKWARD loop for every mini layer from last to first (instead of looping for every angle)

                # 1- Calculate the w_lambda for this lambda (timestep)

########################################                
                w_lambda = np.eye(input_size)
                i = lambda_%2 #top gate of the timestep (with notation j^th gate of the i^th qubit)
                j = int((lambda_ - i)/2) #top gate of the timestep
                
                if j>output_size:
                    j = output_size
                    i = lambda_ - 2*j
                
                while i+j <= input_size-2 and i >= 0 and j >= 0: #Loop over all gates in each timestep: at most O(n/2)
                    if angles_layer_matrix[i][j] != 0: #if the gate's angle is not 0
                        #update incrementally the current matrix of the current timestep,
                        w_lambda = self.gate(i, angles_layer_matrix[i][j], w_lambda) #O(c)
                    #move to the next gate of the current timestep
                    i += 2 #two qubits below
                    j -= 1 #therefore the gate is now closer to the left
########################################     


                # 2- we can now calculate the gradient for each gate in the timestep
########################################             
                i = lambda_%2 #top gate of the timestep (with notation j^th gate of the i^th qubit)
                j = int((lambda_ - i)/2) #top gate of the timestep
                
                if j>output_size:
                    j = output_size
                    i = lambda_ - 2*j
                
                while i+j <= input_size-2 and i >= 0 and j >= 0: #Loop over all gates in each timestep: at most O(n/2)
       
                    angle = angles_layer_matrix[i][j]
                    angle_num = int((i+j)*(i+j+1)/2 + j)
                    #for this gate n°angle_num, we can now calculate dC/dtheta_angle_num :
                    mini_layer_previous = all_mini_layers[lambda_]
                
                    nabla_angle_i = mini_delta[i]*(- np.sin(angle)*mini_layer_previous[i] + np.cos(angle)*mini_layer_previous[i+1]) + mini_delta[i+1]*(- np.cos(angle)*mini_layer_previous[i] - np.sin(angle)*mini_layer_previous[i+1])
                    nabla_angles_list_layer[angle_num] = nabla_angle_i #save gradient

                    #move to the next gate of the current timestep
                    i += 2 #two qubits below
                    j -= 1 #therefore the gate is now closer to the left
########################################     

                    
                # 3- then calculate the mini_delta_lambda at the timestep, by applying the w_lambda transposed to the mini_delta_lambda+1
                mini_delta = w_lambda.T@mini_delta 

            # 4- save this layer's angle gradients    
            nabla_angles[l-1] =  nabla_angles_list_layer #add this (big)layer's angle gradient in the global list
            

        return (nabla_bias, nabla_angles)

    
########################################    
    def calculate_mini_layers(self, layer_input, input_size, output_size, angles_layer_matrix):
        """ NEW - mini layers !
        ### AT THIS "big" LAYER, perform the evolution at each timestep of the current (big) layer and store all the mini layers. 
        ### They correspond to the right slices of the circuit. in O(?)
        ### return a matrix = a list of list : each row is a mini layer = a timestep result """
    
        zeta = np.zeros((2*input_size-2,input_size)) #matrix = list of list: each row is a mini layer = a timestep result = zeta_lambda
        zeta[0] = layer_input.reshape(-1)
        
        for lambda_ in range(2*input_size-3): #Forward loop : from first to last timestep!  #O(2n)
            zeta[lambda_+1] = zeta[lambda_]
            i = lambda_%2 #top gate of the timestep (with notation j^th gate of the i^th qubit)
            j = int((lambda_ - i)/2) #top gate of the timestep
            
            if j>output_size:
                j = output_size
                i = lambda_ - 2*j
            
            # Loop over all gates in each timestep :
            while i+j <= input_size-2 and i >= 0 and j >= 0: #at most O(n/2)
                if angles_layer_matrix[i][j] != 0: #if the gate's angle is not 0
                    #update incrementally the current layer by applying the current timestep
                    zeta[lambda_+1] = self.gate(i, angles_layer_matrix[i][j], zeta[lambda_+1])
                #NOW MOVE TO THE NEXT GATE OF THE CURRENT TIMESTEP
                i += 2 #two qubits below
                j -= 1 #therefore the gate is now closer to the left

        return zeta
######################################## 


############################################################################################################

    def accuracy(self, data, convert=True):
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
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy
    
    def inference(self, data):
        
        outputs = [self.feedforward(x) for (x, y) in data]
        
        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        
#         print("Final Output of the NN:")
#         print(np.round(outputs,3))
#         print("(Predicted, Actual) = ", results)
#         print(sum(int(x == y) for (x, y) in results), "out of", len(data))
       
        return outputs, results

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# def sigmoid(z):
#     """The sigmoid function."""
    
#     return 1.0/(1.0+np.exp(-z))

# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""

#     return sigmoid(z)*(1-sigmoid(z))
