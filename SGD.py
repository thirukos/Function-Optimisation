from copy import deepcopy
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class SGD:
    epsilon = 1e-8
    def __init__(self, f, df, x, stepMethod, parameters, batch_size, trainingData):
        # check given stepMethod
        allowed_stepMethod = ["constant", "polyak", "rmsprop", "heavyball", "adam"]

        if stepMethod not in allowed_stepMethod:
            raise ValueError("Invalid stepMethod, select between constant, polyak, rmspolyak, heavyball, adam")

        self.f = f
        self.df = df
        self.x = deepcopy(x)
        self.n = len(x)
        self.parameters = parameters
        self.batch_size = batch_size
        self.T = trainingData
        self.logs = {
            'x': [deepcopy(self.x)],
            'f': [self.f(self.x, self.T)],
            'step': []
        }
        self.iter_function = self.__get_iteration_function(stepMethod)
        self.__init_function_variables(stepMethod)
    
    #mini-batch calculation
    def minibatch_Iteration(self):

    
        np.random.shuffle(self.T)
        N = len(self.T)
        for i in range(0, N, self.batch_size):
            if i + self.batch_size > N: 
                continue
            sample = self.T[i:(i + self.batch_size)]
            self.iter_function(sample)
        self.logs['x'].append(deepcopy(self.x))
        self.logs['f'].append(self.f(self.x, self.T))

    #calls the respective step size algorithm based on the input parameter
    def __get_iteration_function(self, stepMethod):
        if stepMethod == "constant":
            return self.__constant
        elif stepMethod == "polyak":
            return self.__polyak
        elif stepMethod == "rmsprop":
            return self.__rmsprop
        elif stepMethod == "heavyball":
            return self.__heavy_ball
        else:
            return self.__adam
    
    #initializes required initial variables for the respective step size algorithm
    def __init_function_variables(self, stepMethod):
        if stepMethod == "rmsprop":
            self.logs['step'] = [[self.parameters['alpha0']] * self.n]
            self.vars = {
                'sums': [0] * self.n,
                'alphas': [self.parameters['alpha0']] * self.n
            }
        elif stepMethod == "heavyball":
            self.logs['step'] = [0]
            self.vars = {
                'z': 0
            }
        elif stepMethod == "adam":
            self.logs['step'] = [[0] * self.n]
            self.vars = {
                'ms': [0] * self.n,
                'vs': [0] * self.n,
                'step': [0] * self.n,
                't': 0
            }

    # calculates approximation of partial derivative 
    def __approx_derivative(self, i, sample):
        return sum(self.df[i](*self.x, *sample[j]) for j in range(self.batch_size)) / self.batch_size

    # constant step size implementation
    def __constant(self, sample):
        alpha = self.parameters['alpha']
        for i in range(self.n):
            self.x[i] -= alpha * self.__approx_derivative(i, sample)
        self.logs['step'].append(alpha)

    # polyak step size implementation
    def __polyak(self, sample):
        step = self.f(self.x, sample) / (
            sum(self.__approx_derivative(i, sample) ** 2
            for i in range(self.n)) + self.epsilon
        )
        for i in range(self.n):
            self.x[i] -= step * self.__approx_derivative(i, sample)
        self.logs['step'].append(step)
    
    # rmsprop step size implementation
    def __rmsprop(self, sample):
        alpha0 = self.parameters['alpha0']
        beta = self.parameters['beta']
        alphas = self.vars['alphas']
        sums = self.vars['sums']
        for i in range(self.n):
            self.x[i] -= alphas[i] * self.__approx_derivative(i, sample)
            sums[i] = (beta * sums[i]) + ((1 - beta) * (
                self.__approx_derivative(i, sample) ** 2
            ))
            alphas[i] = alpha0 / ((sums[i] ** 0.5) + self.epsilon)
        self.logs['step'].append(deepcopy(alphas))
    
    # heavyball step size implementation
    def __heavy_ball(self, sample):
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']
        z = self.vars['z']
        z = (beta * z) + (alpha * self.f(self.x, sample) / (sum(
            self.__approx_derivative(i, sample) ** 2
            for i in range(self.n)
        ) + self.epsilon))
        for i in range(self.n):
            self.x[i] -= z * self.__approx_derivative(i, sample)
        self.vars['z'] = z
        self.logs['step'].append(z)
    
    # adam step size implementation
    def __adam(self, sample):
        alpha = self.parameters['alpha']
        beta1 = self.parameters['beta1']
        beta2 = self.parameters['beta2']
        ms = self.vars['ms']
        vs = self.vars['vs']
        step = self.vars['step']
        t = self.vars['t']
        t += 1
        for i in range(self.n):
            ms[i] = (beta1 * ms[i]) + ((1 - beta1) * 
                self.__approx_derivative(i, sample))
            vs[i] = (beta2 * vs[i]) + ((1 - beta2) *
                (self.__approx_derivative(i, sample) ** 2))
            m_hat = ms[i] / (1 - (beta1 ** t))
            v_hat = vs[i] / (1 - (beta2 ** t))
            step[i] = alpha * (m_hat / ((v_hat ** 0.5) + self.epsilon))
            self.x[i] -= step[i]
        self.vars['t'] = t
        self.logs['step'].append(deepcopy(step))

