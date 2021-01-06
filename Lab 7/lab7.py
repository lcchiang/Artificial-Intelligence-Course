# MIT 6.034 Lab 7: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
        return 1
    else:
        return 0


def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+e**(-steepness*(x-midpoint)))


def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0,x)


# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5*(desired_output-actual_output)**2


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    neuron_outputs = {}
    output = net.get_output_neuron()
    
    nodes = net.topological_sort()
    for node in nodes:
        node_sum = 0
        income_nodes = net.get_incoming_neighbors(node)
        for inc_node in income_nodes:
            wire_weight = net.get_wire(inc_node,node).get_weight()
            node_sum += node_value(inc_node,input_values,neuron_outputs) * wire_weight
        neuron_outputs[node] = threshold_fn(node_sum)
        if node == output:
            output_val = neuron_outputs[node]
        
    return (output_val, neuron_outputs)


#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    perturb_inputs = []
    for input_val in inputs:
        new_input_val = [input_val - step_size, input_val, input_val + step_size]
        perturb_inputs.append(new_input_val)
    
    max_output = -INF
    for val0 in perturb_inputs[0]:
        for val1 in perturb_inputs[1]:
            for val2 in perturb_inputs[2]:
                output = func(val0,val1,val2)
                if output > max_output:
                    max_output = output
                    best_combo = [val0,val1,val2]
    return (max_output,best_combo)


def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    start_node = wire.startNode
    end_node = wire.endNode
    
    back_prop_dependencies = set([start_node, wire, end_node])
    if not net.is_output_neuron(end_node):
        out_nodes = net.get_outgoing_neighbors(end_node)
        for out_node in out_nodes:
            wire = net.get_wire(end_node, out_node)
            back_prop_dependencies = back_prop_dependencies.union(get_back_prop_dependencies(net,wire))

    return back_prop_dependencies


def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    nodes = net.topological_sort()
    n = len(nodes)-1
    deltas = {}
    while n >= 0:
        node = nodes[n]
        if net.is_output_neuron(node):
            deltas[node] = (desired_output-neuron_outputs[node])*neuron_outputs[node]*(1-neuron_outputs[node])
        else:
            sum_deltas = 0
            out_nodes = net.get_outgoing_neighbors(node)
            for out_node in out_nodes:
                wire_weight = net.get_wire(node,out_node).get_weight()
                sum_deltas += wire_weight*deltas[out_node]
            deltas[node] = neuron_outputs[node]*(1-neuron_outputs[node])*sum_deltas
        n -= 1
    return deltas


def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    delta_b = calculate_deltas(net,desired_output,neuron_outputs)
    
    nodes = net.topological_sort()
    for node in reversed(nodes):
        inc_nodes = net.get_incoming_neighbors(node)
        for inc_node in inc_nodes:
            wire = net.get_wire(inc_node,node)
            wire_weight_old = wire.get_weight()
            node_val = node_value(inc_node,input_values,neuron_outputs)
            
            delta_wire_weight = r*node_val*delta_b[node]
            wire_weight_new = wire_weight_old + delta_wire_weight
            
            wire.set_weight(wire_weight_new)
    return net


def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    output = forward_prop(net,input_values,sigmoid)
    current_accuracy = accuracy(desired_output, output[0])
    count = 0

    while current_accuracy <= minimum_accuracy:
        net = update_weights(net,input_values,desired_output,output[1],r)
        output = forward_prop(net,input_values,sigmoid)
        current_accuracy = accuracy(desired_output, output[0])
        count += 1
    
    return (net, count)



#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 20
ANSWER_2 = 20
ANSWER_3 = 7
ANSWER_4 = 300
ANSWER_5 = 50

ANSWER_6 = 1
ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small', 'medium','large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A','C']
ANSWER_12 = ['A','E']


#### SURVEY ####################################################################

NAME = 'Luke Chiang'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
