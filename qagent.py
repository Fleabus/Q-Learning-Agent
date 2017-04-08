import tensorflow as tf
import numpy as np
import random as r
from random import randint

'''
    This agent class contains the ability to create a standard q learning neural network using tensorflow
    Features:
        Multilayer neural network
        Q-learning
        Save and reload
        Acts as a superclass to further q-learning algorithms (DQN, AC3, DRQN, DDQN, etc)
'''
class QAgent:
    # Neural Network properties
    x = -1 # inputs
    o = -1 # output weights and biases
    Q = -1 # Q values (predictions)
    allQ = -1 # All the Q values from the last result
    predict = -1 # The highest prediction
    h = []   # Hidden layers
    w = []   # Weights
    nextQ = -1 # The Q value of the next state
    updateModel = -1 # The backpropagation optimizer

    # store Session
    sess = -1

    # Hyperparameters
    lr = 0.1 # Learning Rate
    y = .99 # Long term reward (higher value) | short term reward (lower value)
    e = 0.1  # Exploration rate

    # Layer variables
    input_nodes = -1
    hidden_layer_shape = -1
    output_nodes = -1

    # Example: QAgent(18, [3, 6], 2) - 18 input nodes, 2 hidden layers with 3 and 6 nodes, and 2 output nodes
    def __init__(self, input_nodes, hidden_layer_shape, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_layer_shape = hidden_layer_shape
        self.output_nodes = output_nodes
        # hidden_layer_shape is defined as a vector of the number of nodes on each hidden layer
        # example = [4, 5 ,6 ,8]
        self.x = tf.placeholder('float', [None, input_nodes]) # Input variable
        # Setup hidden layers
        h_index = 0
        prev_nodes = input_nodes
        for hidden_nodes in hidden_layer_shape:
            self.h.append({
                'weights':tf.Variable(tf.random_uniform([prev_nodes, hidden_nodes],0,0.01)),
                #'biases':tf.Variable(tf.random_normal([hidden_nodes]))
            })
            h_index += 1
            prev_nodes = hidden_nodes
        # Setup output layer
        self.output = {
            'weights':tf.Variable(tf.random_uniform([prev_nodes, output_nodes],0,0.01)),
            #'biases':tf.Variable(tf.random_normal([output_nodes]))
        }
        # Setup calculations for predict
        input_values = self.x
        for i in range(len(hidden_layer_shape)):
            input_values = tf.matmul(input_values, self.h[i]['weights'])
            input_values = tf.nn.relu(input_values)
        # Calculate Q values
        self.Q = tf.matmul(input_values, self.output['weights'])
        self.predict = tf.argmax(self.Q, 1)
        # Calculate and minimize loss
        self.nextQ = tf.placeholder(shape=[1,output_nodes], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Q))
        trainer = tf.train.GradientDescentOptimizer(self.lr)
        self.updateModel = trainer.minimize(loss)

    # Initialize variables and assign the session
    def setup(self, sess):
        self.sess = sess
        sess.run(tf.global_variables_initializer())

    # Return an action based on highest q value
    def action(self, s):
        # returns an action an all predicted Q values
        a, self.allQ = self.sess.run([self.predict, self.Q], feed_dict={self.x:s})
        if np.random.rand(1) < self.e:
            a[0] = randint(0, self.output_nodes-1)
        return a

    # Train network using a targetQ value
    def learn(self, a, s, s1, r):
        # Run Q values on new state
        Q1 = self.sess.run(self.Q, feed_dict={self.x:s1})
        maxQ1 = np.max(Q1)
        targetQ = self.allQ.copy()
        # targetQ is all of the new Q values for the next state i.e [0.3, 0.5]
        # Find the action that was performed and update it with new values
        targetQ[0, a[0]] = r + self.y * maxQ1
        #train network using target and predicted values
        self.sess.run([self.updateModel, self.output['weights']], feed_dict={self.x:s, self.nextQ:targetQ})

    # This calculation reduces the exploration value. This is a little too hard coded at the moment
    # Currently requires an iterator(i)
    def update_e(self, i):
        self.e = 1./((i/50) + 10)
