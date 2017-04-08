# Q-Learning-Agent
Classes for implementing Q Learning Agents using Tensorflow.

# Using the QAgent class
The Q-Agent class is a simple implementation of a neural network that predicts Q values mapped to actions. 

It is initialized using the following process:
  - Setup neural network parameters (input, hidden and output layer nodes)
  - Initialize the session and pass into the QAgent setup function
  - Perform an action by passed a state and recieving an action
  - Train model using the 'learn' function
  
# Setup Neural Network Parameters
The nn parameters are defined at instantiation of the object

QAgent(num_of_input_nodes, shape_of_hidden_layers, num_of_output_nodes)

example:
```
agent = QAgent(18, [5, 2], 2)
```
This creates an agent that has 18 input nodes for the environment state to be passed to. It creates two hidden layers with nodes of 5 and 2 respectively and the output layer will have 2 outputs to map to actions.

# Initialize the Session and setup agent
In order to perform Tensorflow functions the agent requires a session. This can be passed through using the agents 'setup(sess)' function

example:
```
with tf.Session() as sess:
    agent.setup(sess)
```
This will intialize all tensorflow variables within the session. If you have two or more agents, the Tensorflow variables will be initialized multiple times for each agent. This does not cause any issues and the session should still be passed to every agent.

# Perform an action
Once a state for the environment is acquired you can pass that into your agents 'action(sstate)' function. This returns the highest Q value for the neural networks outputs as a one dimensional list (although there will only be one value, the action, at position 0. This is to future proof for possible multi-valued actions in more advanced models)

example:
```
agent.action(state)
```
The state needs to be flattened before being passed to the action function and must match the number of input nodes. In order to avoid the agent getting stuck in a local minima, an exploration value of 'e' is defined in the agent. This defines the percentage that the agent will choose a random action. This 'e' value is set to 0.1 as default, but can be set manually (agent.e = 0.05) or be incrementally reduced by calling the 'update_e(i)' function and passing the current iteration. 

# Train the model
Once an action has been performed the agent will need to learn from the result. This uses the agents 'learn(action, state, new_state, reward)'

example:
```
agent.learn(a, s, s1, r)
```
This is doing a few things to update the weights. First, it is calculating a new Q value based on the new state that is passed through. It is then calculating a cost function based on the new Q value and the previous Q value. Using the cost value, backpropagation with gradient descent is actioned to update the weights of the network.


# Features coming
  - Extensions for different Q learning models (Deep Q, Double Q, Dueling Q, Recurrent Q, AS3 etc etc)
  - A memory buffer to store past episodes and train on them
  - The ability to save and load agents
  - Visualization in Tensorboard
  - More examples with different games
  
# For a full example, check out the frozenlake.py demo
