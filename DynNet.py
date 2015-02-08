"""
Dynamic Network, not exactly as described in DeepMind paper.

Combined experience replay technique.

References:
    Recurrent Models of Visual Attention, Google DeepMind, 2014.
"""

from collections import OrderedDict

import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
import theano.sandbox.linalg as LA

import numpy as NP

from Glimpse import glimpse

import cPickle
import sys

_LSTM_tags = ('candt', 'input', 'frget', 'outpt')

T.config.exception_verbosity = 'high'

def identity(x):
    return x

def linear_log(x):
    return NP.log(x + 1) * 0.01 + 0.99 * x

def rectify(x):
    return TT.switch(x > 0, x, 0)

def get_reference(shared):
    return shared.get_value(borrow=True, return_internal_type=True)

def location_normalize(loc, size):
    # normalize location to (-1, 1)
    return loc / size * 2 - 1

def location_restore(loc, size):
    # restore location from (-1, 1) to original coordinates
    return (loc + 1) / 2 * size

def bsl(loc, time):
    return 0.8 * time


class DynNet:
    """
    The overall network for dynamic environment.
    """



    def _add_param(self, name, dims, zero=False):
        # From there on I frequently use a hack to manipulate attributes through __dict__ in order to access them by strings.
        # So basically, self.__dict__["abc"] is to get (or possibly create) the attribute named "abc".
        if zero:
            self.__dict__[name] = T.shared(NP.zeros(dims), name=name)
        else:
            values = NP.random.uniform(-1, 1, dims)
            _, svs, _ = NP.linalg.svd(values)
            self.__dict__[name] = T.shared(values / svs[0], name=name)
        self._params[name] = self.__dict__[name]
        # Add a corresponding shared delta variable.
        # The delta variable should not be a network parameter.
        self.__dict__['d_' + name] = T.shared(NP.zeros(dims), name='d_'+name)
        self._deltas[self._params[name]] = self.__dict__['d_' + name]



    def _add_weights(self, name, input_dims, output_dims, prefix="W_"):
        self._add_param(prefix + name, (output_dims, input_dims))



    def _add_biases(self, name, output_dims, prefix="B_"):
        self._add_param(prefix + name, (output_dims, ), zero=True)



    # Add parameters for fully-connected layers
    def _add_fc(self, name, input_dims, output_dims):
        self._add_weights(name, input_dims, output_dims)
        self._add_biases(name, output_dims)



    # Add parameters for LSTM units
    def _add_LSTM(self):
        for gate_type in _LSTM_tags:
            self._add_weights("LSTM_" + gate_type, self.opts["internal_dims"], self.opts["internal_dims"])
            self._add_weights("LSTM_" + gate_type, self.opts["gn_secnd_dims"], self.opts["internal_dims"], prefix="U_")
            self._add_biases("LSTM_" + gate_type, self.opts["internal_dims"])



    def __init__(self, params=None, **options):
        self.opts = {
        "location_dims" :   2,              # (FIXED) location dimension
        "glimpse_count" :   3,              # (FIXED) number of glimpses taken
        "glimpse_width" :   6,              # (FIXED) width of the smallest or inner-most glimpse
        "gn_first_dims" :   128,            # (MAYBE FIXED) dimension of each part of first layer in the glimpse network (=128)
        "gn_secnd_dims" :   256,            # (MAYBE FIXED) dimension of second layer in the glimpse network (=256)
        "internal_dims" :   256,            # (MAYBE FIXED) dimension of internal state, i.e. number of LSTM units (=256)
        "lstm_out_dims" :   256,            # (MAYBE FIXED) dimension of LSTM output (=256)
        "location_xvar" :   0.005,          # variance of x-coordinate
        "location_yvar" :   0.005,          # variance of y-coordinate
        "actions_count" :   0,              # (FIXED, NOT USED) number of possible actions
        "learning_rate" :   0.005,          # learning rate
        "rate_decay_fn" :   linear_log,     # learning rate decay function r -> r', feel free to change this
        "grad_momentum" :   0.8,            # (NOT USED) gradient momentum, not clear how to use it
        "weight_decays" :   0.05,           # (NOT USED) weight decay, still not clear how to use it
        "training_size" :   200000,         # number of training times
        "minibatch_num" :   20,             # speaks itself, not used since stochastic gradient descent is not implemented yet
        "exp_pool_size" :   50000,          # (NOT USED) size of experience pool, currently all experiences are taken into account.
        "batch_replace" :   None,           # (NOT USED) Replace algorithm, returns index of sample to be replaced
        "additional_fc" :   False,          # Additional fully-connected layers before and after LSTM core, not sure whether it should be added.
        "add_fc_squash" :   NN.sigmoid,     # Squasher function for additional fully-connected layer, no effect if additional_fc is False
        "learningmodel" :   "reinforce_sum",# Learning model.
                                            # 'supervised' -> Supervised model.
                                            # 'reinforce_sum' -> REINFORCE model with costs summed rather than averaged.
        "reinforce_bsl" :   bsl,            # (NOT USED) Baseline function, takes output location and time, returns expectation
        "output_squash" :   identity,       # Location output squasher function
        "save_file_fmt" :   'pickle',       # Parameter save file format, currently only cPickle is supported
        "save_filename" :   'DynNet.pickle',# Save file name
        "load_filename" :   None,           # Load file name.  The trainer loads network from this file if this option is not None.
        "learning_mode" :   'online',       # "online" -> Online learning, "replay" -> Experience replay
        }
        self._params = OrderedDict()
        self._deltas = OrderedDict()

        # Update options first
        for k in options:
            if (k not in self.opts):
                continue
            elif (self.opts[k] != None) and (type(self.opts[k]) is not type(options[k])):
                raise TypeError("Type of option %s is not %s", (k, type(self.opts[k])))
            else:
                self.opts[k] = options[k]
        print 'Current options:'
        for k in self.opts:
            print '\t', k, '\t', self.opts[k]

        print 'Initializing shared variables...'
        self.covariance = T.shared(NP.asarray([[self.opts["location_xvar"], 0], [0, self.opts["location_yvar"]]]), name='cov')

        glm_inputs = (self.opts["glimpse_width"] ** 2) * self.opts["glimpse_count"]

        ### Glimpse Network ###

        # Glimpse network consists of two linear-rectifiers.
        # First linear-rectifier consists of two parts, each processing glimpses and location.
        self._add_fc("glm_in", glm_inputs, self.opts["gn_first_dims"])
        self._add_fc("loc_in", self.opts["location_dims"], self.opts["gn_first_dims"])

        # Second linear-rectifier combines both outputs and transforms the combination.
        self._add_fc("glm_out", 2*self.opts["gn_first_dims"], self.opts["gn_secnd_dims"])

        ### Core Network ###

        # The fully-connected layer to translate output from glimpse network to core network.
        if self.opts["additional_fc"]:
            self._add_fc("trans_in", self.opts["gn_secnd_dims"], self.opts["internal_dims"])

        # Core network is basically an LSTM layer.
        self._add_LSTM()

        # The fully-connected layer to translate output from core network into location network.
        if self.opts["additional_fc"]:
            self._add_fc("trans_out", self.opts["internal_dims"], self.opts["lstm_out_dims"])

        ### Location/Action Network ###

        # The output from LSTM layer is then transferred into a location network, and an additional action network.
        self._add_fc("loc_out", self.opts["lstm_out_dims"], self.opts["location_dims"])
        # Currently only glimpse location is considered, and no other action was taken.  Hence the actions_count must be 0.
        if self.opts["actions_count"] > 0:
            raise AttributeError("option actions_count must be 0")
            self._add_fc("act_out", self.opts["lstm_out_dims"], self.opts["actions_count"])

        # Load network from file if specified
        if self.opts["load_filename"] != None:
            print 'Loading parameters from file...'
            try:
                load_file = open(self.opts["load_filename"], 'rb')
                load_params = cPickle.load(load_file)
                for k in load_params:
                    self.__dict__[k].set_value(load_params[k].get_value())
            except IOError:
                print 'Error opening file %s, ignoring...' % self.opts["load_filename"]

        print 'Initialization complete.'



    def step_lstm(self, location, glimpses, prev_candt, prev_outpt):
        """
        Symbolic representation of LSTM step function.

        @location, @glimpses, @prev_candt are all TensorVariables.

        Parameters:
            location:   A vector representing location.
            glimpses:   A flattened list of glimpse image matrix.
            prev_outpt: Previous outputs of LSTM core.  A vector.
            prev_candt: Previous candidate values of LSTM core.  A vector.

        Returns: (loc_out, next_candt, core_out):
            loc_out:    Chosen location.
            next_candt: Current candidate values of LSTM core.
            core_out:   Current outputs of LSTM core.
        """
        # The network itself is the same between two models.
        # The only difference is cost evaluation.

        ### Glimpse Network ###
        # We flatten both inputs, passing the results into the first linear-rectifier.
        loc_input   = location
        glm_input   = glimpses
        loc_trans   = rectify(TT.dot(self.W_loc_in, loc_input) + self.B_loc_in)
        glm_trans   = rectify(TT.dot(self.W_glm_in, glm_input) + self.B_glm_in)

        # Both outputs are then combined and processed by the second linear-rectifier.
        glm_immdt   = TT.concatenate([loc_trans, glm_trans])
        glm_out     = rectify(TT.dot(self.W_glm_out, glm_immdt) + self.B_glm_out)

        ### Core Network ###
        # Output from glimpse network is first transformed into core network input by a fully-connected layer.
        if self.opts["additional_fc"]:
            core_in = self.opts["add_fc_squash"](TT.dot(self.W_trans_in, glm_out) + self.B_trans_in)
        else:
            core_in = glm_out
        
        # The core network input is then processed together with the previous candidate state.
        # Calculate activation of each gate first.
        candt_act   = TT.tanh   (   TT.dot(self.W_LSTM_candt, core_in)
                                  + TT.dot(self.U_LSTM_candt, prev_outpt)
                                  + self.B_LSTM_candt)
        input_act   = NN.sigmoid(   TT.dot(self.W_LSTM_input, core_in)
                                  + TT.dot(self.U_LSTM_input, prev_outpt)
                                  + self.B_LSTM_input)
        frget_act   = NN.sigmoid(   TT.dot(self.W_LSTM_frget, core_in)
                                  + TT.dot(self.U_LSTM_frget, prev_outpt)
                                  + self.B_LSTM_frget)
        outpt_act   = NN.sigmoid(   TT.dot(self.W_LSTM_outpt, core_in)
                                  + TT.dot(self.U_LSTM_outpt, prev_outpt)
                                  + self.B_LSTM_outpt)
        # Cell output (or next candidate value) is obtained from adding gated input and forgotten memory together.
        gated_input = input_act * candt_act
        forgot_mem  = frget_act * prev_candt
        next_candt  = gated_input + forgot_mem
        # Gating squashed result returns the LSTM output
        core_out    = outpt_act * TT.tanh(next_candt)

        # Core output is then transformed by a fully-connected layer to the location network
        if self.opts["additional_fc"]:
            loc_in  = self.opts["add_fc_squash"](TT.dot(self.W_trans_out, core_out) + self.B_trans_out)
        else:
            loc_in  = core_out

        ### Location network ###
        # No squashing is applied to the location network output.
        # Only a simple linear transformation is done.
        loc_out     = self.opts["output_squash"](TT.dot(self.W_loc_out, loc_in) + self.B_loc_out)

        return loc_out, next_candt, core_out



    def step_supervised(self, location, glimpses, real_location, prev_candt, prev_outpt):
        """
        Cost function for LSTM in supervised model.
        """
        loc_out, next_candt, core_out = self.step_lstm(location, glimpses, prev_candt, prev_outpt)
        loc_diff = loc_out - real_location
        cost = TT.dot(loc_diff, loc_diff)
        return loc_out, next_candt, core_out, cost



    def step_reinforce(self, location, glimpses, step_num, chosen_loc, reward, prev_candt, prev_outpt):
        """
        Cost function for LSTM in REINFORCE model.
        """
        loc_out, next_candt, core_out = self.step_lstm(location, glimpses, prev_candt, prev_outpt)
        loc_diff = chosen_loc - loc_out
        # Log-probability density function for independent bivariate normal distribution, with coefficients discarded.
        # It is essentially the distance between chosen location and mean location...
        pdf = TT.dot(loc_diff, loc_diff)
        # REINFORCE is an acronym of "REward Increment = Nonnegative Factor * Offset Reinforcement * Characteristic Eligibility",
        # Since "Nonnegative Factor" there is learning rate here, and we can view the reward increment as a gradient ascent step of a particular cost function.
        # Moreover, the "Characteristic Eligibility" is exactly the differentiation of probability density function.
        # Hence, we can write out that cost function.
        # NOTE: The only difference between the so-called reinforcement learning method with traditional supervised method is now the granularity of cost function.
        #       Did I miss something?
        #       References: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, R. J. Williams, 1992
        # NOTE: The baseline I choose is just the previous cumulative reward.
        #       Perhaps it's easy to implement because the stepping function only need *current* reward instead of *cumulative* reward.
        #       I still can't figure out how to calculate expectation of current reward.  Maybe I need another network or something for approximation...?
        #       References: Policy Gradient Methods for Reinforcement Learning with Function Approximation, R. S. Sutton et al., NIPS 1999.
        cost = pdf * reward
        return loc_out, next_candt, core_out, cost



    def replay(self):
        """
        Experience replay and Stochastic Gradient Descent optimization (?)
        """
        d = {}
        # Shuffle... then take the first few samples to do minibatch.
        NP.random.shuffle(self.exp_pool)
        batch_num = min(self.opts["minibatch_num"], len(self.exp_pool))
        if batch_num == 0:
            return
        for i in range(0, batch_num):
            if self.opts["learningmodel"] == 'supervised':
                loc_in, glm_in, real_out = self.exp_pool[i]
                self.learn_func(loc_in, glm_in, real_out)
            else:
                loc_in, glm_in, time, chosen_loc, reward = self.exp_pool[i]
                self.learn_func(loc_in, glm_in, time, chosen_loc, reward)
            for param in self._params:
                d[param] = d.get(param, 0) + self._deltas[self._params[param]].get_value()
        for param in self._params:
            self._params[param].set_value(self._params[param].get_value() - d[param] / batch_num)



    def online(self):
        """
        Online learning method
        """
        d = {}
        if self.opts["learningmodel"] == 'supervised':
            loc_in, glm_in, real_out = self.exp_pool[-1]
            self.learn_func(loc_in, glm_in, real_out)
        else:
            loc_in, glm_in, time, chosen_loc, reward = self.exp_pool[-1]
            self.learn_func(loc_in, glm_in, time, chosen_loc, reward)
        for param in self._params:
            d[param] = self._deltas[self._params[param]].get_value()
        for param in self._params:
            self._params[param].set_value(self._params[param].get_value() - d[param])



    def train(self, env):
        """
        Train the network within given environment @env.
        """

        # Setup step function.
        print 'Setting up step functions...'
        sym_loc_in = TT.vector('loc_in')
        sym_glm_in = TT.vector('glm_in')
        sym_prev_candt = TT.vector('prev_candt')
        sym_prev_outpt = TT.vector('prev_outpt')
        sym_loc_out, sym_next_candt, sym_next_outpt = self.step_lstm(sym_loc_in, sym_glm_in, sym_prev_candt, sym_prev_outpt)
        step_func = T.function([sym_loc_in, sym_glm_in, sym_prev_candt, sym_prev_outpt], [sym_loc_out, sym_next_candt, sym_next_outpt])
        
        # Calculate mean cost for all steps.
        # Suppose we have a list of inputs here, represented as symbols...
        print 'Setting up cost function...'
        sym_loc_in_list = TT.matrix('loc_in_l')
        sym_glm_in_list = TT.matrix('glm_in_l')
        sym_candt_list = TT.matrix('candt_l')
        sym_outpt_list = TT.matrix('outpt_l')
        if self.opts["learningmodel"] == 'supervised':
            sym_real_loc_list = TT.matrix('real_loc_l')
            # I have to admit that theano.scan() function is quite obscure.
            # In a word, theano.scan() repeatedly invokes a function @fn, producing outputs.
            # @fn sequentially takes input from lists specified in @sequences, and previous outputs specified in @outputs_info.
            # The function output matches @outputs_info.
            # theano.scan() returns a sequence of outputs generated by @fn.
            # So it's relatively easy to write RNNs using theano.scan(), but learning how to use it is quite difficult.
            step_scan, _ = T.scan(fn=self.step_supervised,
                                  sequences=[sym_loc_in_list, sym_glm_in_list, sym_real_loc_list],
                                  outputs_info=[None, TT.zeros((self.opts["internal_dims"],)), TT.zeros((self.opts["lstm_out_dims"],)), None])
            sym_cost = step_scan[3].mean()
        else:
            sym_steps = TT.iscalar('steps')
            sym_step_num_list = TT.arange(sym_steps)
            sym_chosen_loc_list = TT.matrix('chosen_loc_l')
            sym_reward_list = TT.vector('reward_l')
            step_scan, _ = T.scan(fn=self.step_reinforce,
                                  sequences=[sym_loc_in_list, sym_glm_in_list, sym_step_num_list, sym_chosen_loc_list, sym_reward_list],
                                  outputs_info=[None, TT.zeros((self.opts["internal_dims"],)), TT.zeros((self.opts["lstm_out_dims"],)), None])
            sym_cost = step_scan[3].sum()
        sym_cost.name = 'cost'

        # Calculate gradients.
        print 'Building gradients...'
        #sym_grads = {}
        #for param in self._params.values():
        #    print '\tBuilding gradient for %s...' % param.name
        #    sym_grads[param] = T.grad(sym_cost, param)
        #print '\tBuilding overall gradients...'
        sym_grads = T.grad(sym_cost, self._params.values())

        print 'Setting up update model...'
        updates = OrderedDict()
        sym_learn_rate = T.shared(self.opts["learning_rate"], name='learn_rate')
        # Only deltas are updated by learning function.  Real action is taken in replay() function.
        for i, param in enumerate(self._params.values()):
            # Currently I'm not considering adding momentum and weight decays into gradient increment.
            updates[self._deltas[param]] = sym_learn_rate * sym_grads[i]

        # Create learning functions for adjusting weights.
        if self.opts["learningmodel"] == 'supervised':
            self.learn_func = T.function([sym_loc_in_list, sym_glm_in_list, sym_real_loc_list],
                                         sym_cost,
                                         updates=updates,
                                         on_unused_input='ignore')
        else:
            self.learn_func = T.function([sym_loc_in_list, sym_glm_in_list, sym_steps, sym_chosen_loc_list, sym_reward_list],
                                         sym_cost,
                                         updates=updates,
                                         on_unused_input='ignore')

        # Define experience pool for replaying experience (and doing SGD).
        self.exp_pool = []

        # Determine update method according to learning mode
        if self.opts["learning_mode"] == 'online':
            updater = self.online
        elif self.opts["learning_mode"] == 'replay':
            updater = self.replay

        # Start training
        print 'TRAINING START!'
        try:
            mean = 0.
            prev_mean = 0.
            sum_rwd = 0
            sum_step = 0
            effective_gamenum = 0
            for gamenum in xrange(0, self.opts["training_size"]):
                # OK, here's how I intend to do it:
                # Each time we train the network, we let the agent play an entire game, recording the choices and feedbacks into a sequence.
                # Then we could feed-forward and back-propagate the network using that sequence by ordinary theano.scan() and theano.grad().

                # Start a game
                env.start()

                # Initialize training sequences
                loc_in = [NP.zeros((self.opts["location_dims"],))]
                orig_glm_in = [NP.zeros((self.opts["glimpse_width"], self.opts["glimpse_width"], self.opts["glimpse_count"]))]
                glm_in = [NP.zeros((self.opts["glimpse_width"] * self.opts["glimpse_width"] * self.opts["glimpse_count"]))]
                loc_out = [NP.zeros((self.opts["location_dims"],))]
                core_out = [NP.zeros((self.opts["lstm_out_dims"],))]
                candt = [NP.zeros((self.opts["internal_dims"],))]
                real_out = [location_normalize(NP.array([env._ball.posX, env._ball.posY]), env.size())]
                reward = [0.]
                chosen_loc = [NP.zeros((self.opts["location_dims"],))]
                cost = [0.]
                time = 0
                rwd = 0

                # Randomly choose a starting point
                loc_in.append(NP.random.uniform(-1, 1, 2))

                # Keep tracking (randomly?) until the ball leaves the screen
                while not env.done():
                    # Fetch glimpses
                    orig_glm_in.append(glimpse(env.M, self.opts["glimpse_width"], location_restore(loc_in[-1], env.size())))
                    glm_in.append(NP.asarray(orig_glm_in[-1]).flatten())
                    # Pass the glimpses, location, along with previous LSTM states into the step function ONCE.
                    lo, ca, co = step_func(loc_in[-1], glm_in[-1], candt[-1], core_out[-1])
                    # Record the states and choice.
                    candt.append(ca)
                    core_out.append(co)
                    loc_out.append(lo)

                    # Step over
                    time += 1
                    env.tick()

                    # Calculate informations (labels) available for determining the cost later.
                    # Meanwhile, select next location input.
                    if self.opts["learningmodel"] == 'supervised':
                        # In supervised model, the next location the agent will choose is equal to the location network output.
                        # No distribution is applied.
                        loc_in.append(lo)
                        chosen_loc.append(lo)
                        # The locations stored by agent is zoomed to (-1, -1) ~ (1, 1).
                        ro = location_normalize(NP.array([env._ball.posX, env._ball.posY]), env.size())
                        real_out.append(ro)
                        rwd = (1 if env.is_tracking(location_restore(lo, env.size()), self.opts["glimpse_width"]) else 0)
                        reward.append(rwd)
                        #print '\ttime=', time
                        #print '\t\tloc_out =', loc_out[-1]
                        #print '\t\treal_out=', real_out[-1]
                    else:
                        # In reinforcement learning model, the location picked by agent follows an independent bivariate normal distribution.
                        cl = NP.random.multivariate_normal(lo, self.covariance.get_value())
                        chosen_loc.append(cl)
                        loc_in.append(cl)
                        rwd = (1 if env.is_tracking(location_restore(cl, env.size()), self.opts["glimpse_width"]) else 0)
                        reward.append(rwd)
                        ro = location_normalize(NP.array([env._ball.posX, env._ball.posY]), env.size())
                        real_out.append(ro)
            
                # Remove trailing loc_in element first.
                loc_in.pop()
                time -= 1

                # Check cost and add history into experience pool.
                if self.opts["learningmodel"] == 'supervised':
                    c = self.learn_func(loc_in, glm_in, real_out)
                    mean = (mean * gamenum + c) / (gamenum + 1)
                    print '\x1b[31m' if mean > prev_mean else '\x1b[32m', 'Game #%d' % gamenum, '\tCost: %.10f' % c, '\tMean: %.10f' % mean, \
                            '\tSteps: %d' % time, '\x1b[37m'
                    prev_mean = mean
                    self.exp_pool.append((loc_in, glm_in, real_out))
                else:
                    c = self.learn_func(loc_in, glm_in, time + 1, chosen_loc, reward)
                    sum_rwd += sum(reward)
                    sum_step += time
                    mean_prob = float(sum_rwd) / sum_step
                    # Put the game experience into the pool if we're using online learning method, or the game is effective and we're replaying experiences.
                    if ((sum(reward) != 0) and (self.opts["learning_mode"] == 'replay')) or (self.opts["learning_mode"] == 'online'):
                        self.exp_pool.append((loc_in, glm_in, time + 1, chosen_loc, reward))
                    
                    if sum(reward) != 0:
                        mean = (mean * effective_gamenum + c) / (effective_gamenum + 1)
                        effective_gamenum += 1
                    print '\x1b[0;31m' if mean_prob > prev_mean else '\x1b[0;32m', \
                            'Game #%d' % gamenum, '\tEffective #%d' % effective_gamenum, '\tCost: %.10f' % c, '\tMean: %.10f' % mean, \
                            '\tReward: %d' % sum(reward), '\tSteps: %d' % time, '\tProb: %.10f' % (sum(reward) / time), \
                            'Mean Prob: %.10f' % mean_prob, '\x1b[0;37m'
                    prev_mean = mean_prob

                # Replay experiences.
                updater()

                if (gamenum % 100 == 0):
                    print 'Step #\t\tloc_out\t\t\t\tchosen_loc\t\t\treward\treal_out\t\t\t\tdistance\tchosen_dist'
                    for t in range(0, time):
                        loc_out_r = location_restore(loc_out[t], env.size())
                        chosen_loc_r = location_restore(chosen_loc[t], env.size())
                        real_out_r = location_restore(real_out[t], env.size())
                        print t, '\t\t', loc_out_r, '\t', chosen_loc_r, '\t',\
                                reward[t], '\t', real_out_r, '\t', \
                                NP.sqrt(NP.dot(real_out_r - loc_out_r, real_out_r - loc_out_r)) if self.opts["learningmodel"] == 'supervised' \
                                else NP.max(NP.abs(real_out_r - loc_out_r)), \
                                '\t', NP.sqrt(NP.dot(chosen_loc_r - real_out_r, chosen_loc_r - real_out_r)) if self.opts["learningmodel"] == 'supervised' else \
                                NP.max(NP.abs(real_out_r - chosen_loc_r))
                
                # Decrease learning rate
                sym_learn_rate.set_value(self.opts["rate_decay_fn"](sym_learn_rate.get_value()))
                                
        except (KeyboardInterrupt, IOError):
            sys.stderr.write('Interrupt signal caught, saving network parameters...\n')
            if self.opts["save_filename"] != None:
                save_file = open(self.opts["save_filename"], 'wb')
                cPickle.dump(self._params, save_file, protocol=2)
                save_file.close()
            else:
                sys.stderr.write('Save file not specified, ignoring...\n')
