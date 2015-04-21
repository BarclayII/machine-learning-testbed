
import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.sandbox.linalg as LA

import numpy as NP
import numpy.random as RNG

import cPickle

from collections import OrderedDict



# For smooth migration to GPU.
#
# Some GPUs only support 32-bit floats.  This function casts arrays into
# a 'floatX' type, which could be configured in THEANO_FLAGS environment
# variable.
def to_floatX_array(x):
    return NP.asarray(x, dtype=T.config.floatX)

def to_floatX_scalar(x):
    return NP.__dict__[T.config.floatX](x)



# Define a new theano shared tensor or scalar variable with default float type.
def new_shared_array(x, name=None):
    return T.shared(to_floatX_array(x), name=name)

def new_shared_scalar(x, name=None):
    return T.shared(to_floatX_scalar(x), name=name)



# Identity function
def identity(x):
    return x

# Rectifier function
def rectify(x):
    return TT.switch(x > 0, x, 0)



# Base class of all neural network layers.
class NetLayer:
    '''
    Base class of all neural network layers.
    '''
    # A helper function which creates the parameter list __params
    # if necessary.
    def _check_param_list(self):
        if '_params' not in self.__dict__:
            self._params = OrderedDict()



    # Add a network parameter with given name and dimensions
    def _add_param(self, name, dims, zero=False):
        if not zero:
            values = RNG.uniform(-1, 1, dims)
            _, svs, _ = NP.linalg.svd(values)
            self.__dict__[name] = new_shared_array(values / svs[0], name)
        else:
            self.__dict__[name] = new_shared_array(NP.zeros(dims), name)

        self._check_param_list()

        self._params[name] = self.__dict__[name]



    def _add_scalar_param(self, name):
        self.__dict__[name] = new_shared_scalar(RNG.ranf(), name)
        self._check_param_list()
        self._params[name] = self.__dict__[name]



    # Get the parameter list
    def get_params(self):
        '''
        Returns parameter list of this layer.
        '''
        self._check_param_list()

        return self._params



    # Function for processing input to output
    def _step(self, **kwargs):
        '''
        Processes @input_data through this layer.

        Returns a list of outputs in the following format:
        [[list_of_immediate_result],
         [list_of_1_tap_recurrent_result],
         [list_of_2_tap_recurrent_result],
         ...]
        '''
        pass



    def _step_batch(self, **kwargs):
        pass



    def __init__(self, name, **kwargs):
        self.name = name



class FullyConnectedLayer(NetLayer):
    '''
    Fully connected layer class.

    A fully connected layer usually consists of a weight matrix, a bias
    vector, and an optional non-linear activation function.
    '''
    def __init__(self, name, input_dims, output_dims, act_func=None):
        NetLayer.__init__(self, name)
        self._add_param('W_'+name, (input_dims, output_dims))
        self._add_param('B_'+name, (output_dims,), zero=True)
        # Aliases
        self.W = self.__dict__['W_'+name]
        self.B = self.__dict__['B_'+name]
        self._act_func = identity if act_func == None else act_func



    def _step(self, input_data):
        return self._act_func(TT.dot(input_data, self.W) + self.B)



class SigmoidFullyConnectedLayer(FullyConnectedLayer):
    '''
    Fully connected layer with sigmoid (logistic) function as activation
    function.
    Derived from FullyConnectedLayer class.
    '''
    def __init__(self, name, input_dims, output_dims):
        FullyConnectedLayer.__init__(self, name, input_dims, output_dims, NN.sigmoid)



class RectifiedFullyConnectedLayer(FullyConnectedLayer):
    '''
    Fully connected layer with rectifier function as activation function.
    Derived from FullyConnectedLayer class.
    Alias: ReLU
    '''
    def __init__(self, name, input_dims, output_dims):
        FullyConnectedLayer.__init__(self, name, input_dims, output_dims, rectify)
# An alias of RectifiedFullyConnectedLayer
ReLU = RectifiedFullyConnectedLayer



class SoftmaxFullyConnectedLayer(FullyConnectedLayer):
    '''
    Fully connected layer with softmax as activation function.
    Derived from FullyConnectedLayer class.
    '''
    def __init__(self, name, input_dims, output_dims):
        FullyConnectedLayer.__init__(self, name, input_dims, output_dims, NN.softmax)



class LSTMLayer(NetLayer):
    '''
    An LSTM layer.

    Usually a fully connected layer is placed before and after the LSTM layer
    for dimension conversion.
    '''
    def __init__(self, name, nr_units, **kwargs):
        '''
        @kwargs could contain key-value pairs with key
            state_act_func (default tanh)
            input_act_func (default sigmoid)
            frget_act_func (default sigmoid)
            outpt_act_func (default sigmoid)
            act_func       (default tanh)
        for specifying activation functions.
        '''
        NetLayer.__init__(self, name)
        # The four types of gates
        self.__gate_type = ('state', 'input', 'frget', 'outpt')
        # Default activation function setup
        self._state_act_func = TT.tanh
        self._input_act_func = NN.sigmoid
        self._frget_act_func = NN.sigmoid
        self._outpt_act_func = NN.sigmoid
        self._act_func       = TT.tanh
        # Add parameters for each type of gate.
        for gate in self.__gate_type:
            param_name = 'W_' + gate + '_' + name
            param_alias = 'W_' + gate
            self._add_param(param_name, (nr_units, nr_units))
            self.__dict__[param_alias] = self.__dict__[param_name]
            if gate + '_act_func' in kwargs:
                # self._state_act_func = kwargs['state_act_func'] etc.
                self.__dict__['_'+gate+'_act_func'] = kwargs[gate+'_act_func']
        if 'act_func' in kwargs:
            self._act_func = kwargs['act_func']



    def _step(self, input_data, recur_inputs):
        '''
        Receives the input, as well as previous state and output activation.
        Returns a tuple consisting of next state and output activation.

        @input_data must be a vector with dimension equal to number of units
        in this layer.
        '''
        prev_state, prev_outpt = recur_inputs
        state_act   = self._state_act_func(TT.dot(prev_outpt, self.W_state) + input_data)
        input_act   = self._input_act_func(TT.dot(prev_outpt, self.W_input) + input_data)
        frget_act   = self._frget_act_func(TT.dot(prev_outpt, self.W_frget) + input_data)
        outpt_act   = self._outpt_act_func(TT.dot(prev_outpt, self.W_outpt) + input_data)
        gated_input = input_act * state_act
        forgot_mem  = frget_act * prev_state
        next_state  = gated_input + forgot_mem
        next_outpt  = outpt_act * self._act_func(next_state)

        return next_state, next_outpt



class NeuralNetwork(NetLayer):
    def _check_layer_list(self):
        if 'layers' not in self.__dict__:
            self.layers = OrderedDict()



    def add_layer(self, layer):
        self._check_param_list()
        self._check_layer_list()

        layer_params = layer.get_params()
        for param in layer_params:
            self._params[param] = layer_params[param]
        self.layers[layer.name] = layer



    def __init__(self, filename=None):
        if filename != None:
            try:
                f = open(filename, "rb")
                self.layers, self._params = cPickle.load(f)
            except IOError:
                pass



    def save(self, filename):
        try:
            f = open(filename, "wb")
            cPickle.dump((self.layers, self._params), f, protocol=2)
        except:
            print 'Cannot save parameter'



class TrajectoryRNN(NeuralNetwork):
    '''
    The overall network goes like this:
    input_part1 input_part2 input_part3 ...
         |           |           |
         v           v           v
       ReLU        ReLU        ReLU
         |           |           |
         -------------------------
                     |
                     v
                   ReLU
                     |
                     v
                   LSTM0
                     |
                     v
                    FC0
                     |
                     v
                   LSTM1
                     |
                     v
                    ...
                     |
                     v
                   LSTMn
                     |
                     v
                   ReLU
                     |
         -------------------------
         |           |           |
         v           v           v
      Softmax     Softmax     Softmax
         |           |           |
         v           v           v
    out_part1   out_part2   out_part3   ...
    '''
    def __init__(self, input_parts, nr_rnn_units, output_parts, filename=None):
        self.nr_input_parts = len(input_parts)
        self.nr_rnn_layers = len(nr_rnn_units)
        self.nr_output_parts = len(output_parts)
        self.nr_rnn_units = nr_rnn_units
        for i, dim in enumerate(input_parts):
            self.add_layer(RectifiedFullyConnectedLayer('input' + str(i), dim, 128))
        self.add_layer(FullyConnectedLayer('lstm_in0', 128 * len(input_parts), nr_rnn_units[0]))
        for i, dim in enumerate(nr_rnn_units):
            self.add_layer(LSTMLayer('lstm' + str(i), nr_rnn_units[i]))
            if i != self.nr_rnn_layers - 1:
                self.add_layer(FullyConnectedLayer('lstm_in' + str(i+1), nr_rnn_units[i], nr_rnn_units[i+1]))
        self.add_layer(RectifiedFullyConnectedLayer('lstm_out', nr_rnn_units[-1], 128 * len(input_parts)))
        for i, dim in enumerate(output_parts):
            self.add_layer(SoftmaxFullyConnectedLayer('out' + str(i), 128 * len(input_parts), dim))

        NeuralNetwork.__init__(self, filename)

        input_list = [TT.vector('in'+str(i)) for i in range(0, self.nr_input_parts)]
        rec_state_list = [TT.vector('rec'+str(i)) for i in range(0, self.nr_rnn_layers * 2)]

        step_input_list = input_list + rec_state_list
        self.step = T.function(step_input_list, self._step(*step_input_list))

        input_seq_list = [TT.matrix('inseq'+str(i)) for i in range(0, self.nr_input_parts)]
        truth_seq_list = [TT.matrix('truth'+str(i)) for i in range(0, self.nr_output_parts)]
        scan_input_list = input_seq_list + truth_seq_list
        self.inputs = scan_input_list
        self.outputs = self._scan(*scan_input_list)
        self.step_scan = T.function(self.inputs, self.outputs)

        self._grad_array = T.grad(self._scan(*scan_input_list)[-1].mean(), self._params.values())
        self._grad = OrderedDict()
        for i, param in enumerate(self._params.values()):
            self._grad[param] = self._grad_array[i]
        self.g = OrderedDict()
        for param in self._params.values():
            self.g[param] = new_shared_array(NP.zeros_like(param.get_value()))
        updates = OrderedDict()
        for param in self._params.values():
            updates[self.g[param]] = self._grad[param]
        self.grad = T.function(self.inputs, self._grad_array, updates=updates)



    def _step(self, *args):
        '''
        The parameters are in the following order:
        First len(input_parts) parameters are network's input.
        Last len(nr_rnn_units) * 2 parameters are recurrent states.
        '''
        inputs = args[: self.nr_input_parts]
        rec_in = args[-self.nr_rnn_layers * 2 :]
        first_outputs = []

        for i in range(0, self.nr_input_parts):
            first_outputs.append(self.layers['input'+str(i)]._step(inputs[i]))

        lstm_in = TT.concatenate(first_outputs)
        rec_states = []
        for i in range(0, self.nr_rnn_layers):
            lstm_imm = self.layers['lstm_in'+str(i)]._step(lstm_in)
            lstm_out = self.layers['lstm'+str(i)]._step(lstm_imm, rec_in[i * 2 : (i + 1) * 2])
            lstm_in = lstm_out[1]
            rec_states += lstm_out

        mmt_in = self.layers['lstm_out']._step(lstm_in)

        outputs = []
        for i in range(0, self.nr_output_parts):
            outputs.append(self.layers['out'+str(i)]._step(mmt_in))

        return tuple(outputs + rec_states)



    def _step_with_cost(self, *args):
        inputs = args[: self.nr_input_parts]
        rec_in = args[-self.nr_rnn_layers * 2 :]
        truth = args[self.nr_input_parts : -self.nr_rnn_layers * 2]
        step_out = self._step(*(inputs + rec_in))
        outputs = step_out[: self.nr_output_parts]
        cost = 0
        for i in range(0, self.nr_output_parts):
            diff = outputs[i] - truth[i]
            cost += TT.dot(diff, TT.transpose(diff))
        return step_out + (cost,)



    def _scan(self, *args):
        input_seqs = args[: self.nr_input_parts]
        truth_seqs = args[-self.nr_output_parts :]
        rec_states = []
        for dim in self.nr_rnn_units:
            rec_states += [TT.zeros((dim,)), TT.zeros((dim,))]
        sc, _ = T.scan(fn = self._step_with_cost,
                       sequences = input_seqs + truth_seqs,
                       outputs_info = [None] * self.nr_output_parts  # Outputs
                                    + rec_states                     # Recurrent states
                                    + [None])                        # Cost
        return sc



class Trainer:
    def __init__(self, network, learn_rate = 0.0):
        self.network = network
        self.learn_rate = new_shared_scalar(learn_rate)
        self.learn = None



class OnlineTrainer(Trainer):
    def __init__(self, network, learn_rate, rate_decay_func=identity):
        Trainer.__init__(self, network, learn_rate)

        updates = OrderedDict()

        for param in self.network._params.values():
            updates[param] = param - self.learn_rate * self.network._grad[param]
        updates[self.learn_rate] = rate_decay_func(self.learn_rate)

        self.learn = T.function(self.network.inputs, self.network.outputs, updates=updates)



class RMSpropTrainer(Trainer):
    def __init__(self, network, learn_rate = 0.05, decay_rate = 0.95, momentum = 0.9, regularizer = 1e-4):
        Trainer.__init__(self, network, learn_rate)
        self.network = network
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.regularizer = regularizer

        self._f = OrderedDict()
        self._g = OrderedDict()
        self._delta = OrderedDict()

        for param in self.network._params.values():
            self._f[param] = new_shared_array(NP.zeros_like(param.get_value()))
            self._g[param] = new_shared_array(NP.zeros_like(param.get_value()))
            self._delta[param] = new_shared_array(NP.zeros_like(param.get_value()))

        updates = OrderedDict()
        for param in self.network._params.values():
            updates[self._f[param]] = self._f[param] * self.decay_rate + (1 - self.decay_rate) * self.network._grad[param]
            updates[self._g[param]] = self._g[param] * self.decay_rate + (1 - self.decay_rate) * (self.network._grad[param] ** 2)
            updates[self._delta[param]] = self._delta[param] * self.momentum \
                                        - self.learn_rate / TT.sqrt(self._g[param] - self._f[param] ** 2 + self.regularizer) * self.network._grad[param]
            updates[param] = param + self._delta[param]

        self.learn = T.function(self.network.inputs, self.network.outputs, updates=updates)
