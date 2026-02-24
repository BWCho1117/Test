
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import logm, expm

class HiddenMarkovModel():

    def __init__ (self, process = 'Gaussian'):
        if process == 'Gaussian':
            self._P = 'G'

        # dictionary of model states
        self._S = {}
        self._ct = 0

        # matrix of transition probabilities
        self._M = None

    def add_state (self, **kwargs):
        '''
        adds a state to the Markov chain, with arbitrary parameters,
        for example mean and std for a Gaussian process
        '''
        D = {}
        for k in kwargs.keys():
            D[k] = kwargs[k]
        self._S[str(self._ct)] = D
        self._ct += 1

    def add_state_list (self, S, S_pars):
        '''
        S is a list containing parameters (name, mean, std, etc) for each state.
        S_pars contains the names of the parameters for each state
        '''
        D = {}

        for states in S:
            name = str(self._ct)            
            D[name] = {}
            for par in S_pars:
                D[name][par] = states[S_pars.index(par)]
            self._ct += 1

        self._S = D

    def add_transition_prob_matrix (self, M, probe_duration=1):
        # The probability transition matrix is measured
        # overy a time described by probe_duration
        # We compute the rate matrix as the logarithm of M

        self._M_t_wndw = probe_duration
        self._M_measured = M
        self._M = M
        
        # rate matrix for the corresponding continuous-time Markov model
        self._R = logm(M)/self._M_t_wndw

    def initialise (self, state):
        self._S0 = state

    def compute_jump_matrix_from_rates (self, t):
        M = expm(self._R*t)

        # We check that the transition matrix makes sense
        # (i.e. the proabbilities in each row sum to 1).
        # If not, we flag it and normalize each row to 1
        r, c = np.shape(M)
        for i in range(r):
            p = np.sum(M[i,:])
            if (p != 1):
                print ("Jump matrix total probability: ", p)
                M[i, :] = M[i, :]/p
        self._M = M
        return M

    def _lorentzian (self, x, w):
        P = 1/(1 + (x/(w/2))**2)
        return P

    def generate (self, nr_steps):
        '''
        Generates a trace with a given number of time steps (nr_steps)
        '''

        state = self._S0
        values = np.zeros(nr_steps)
        sss = np.zeros(nr_steps)
        for i in range(nr_steps):
            sss[i] = self._S[str(state)]['mu']
            values[i] = np.random.normal (self._S[str(state)]['mu'], self._S[str(state)]['sigma'])
            state = np.random.choice (range(self._ct), 1, p=self._M[state, :])[0]

        return values, sss

