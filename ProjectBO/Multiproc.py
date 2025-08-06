#Bayesian Optimization - Reduced Forecast
#credit - https://machinelearningmastery.com/what-is-bayesian-optimization/
import multiprocessing
import LSTMproject

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor

from warnings import catch_warnings
from warnings import simplefilter

from copy import deepcopy as dc
import random #for setting the seed
import math #for error analysis (square root function)

def runfile(look, hid, re, lr, epoch, bat, dat): #define function to run LSTM code
    data=LSTMproject.arrange(look) #rearrange dataset
    ferr=LSTMproject.LSTMmodel(look, hid, re, lr, epoch, bat, dat, data) #generate LSTM and calculate error

    return ferr #return error

def ran(stacks): #generate random input values
  look=random.randrange(50, 450)
  hid=random.randrange(1, 30)
  lr=random.randrange(1, 999)
  epoch=random.randrange(40, 100)
  bat=random.randrange(16, 256)
  dat=random.randrange(70, 95)
  if (stacks==0):
     re=random.randrange(1, 4)
  else:
    re=stacks

  inputs=(look, hid, re, lr, epoch, bat, dat)
  return inputs

def surrogate(model, X): # surrogate or approximation for the objective function
	with catch_warnings(): # catch any warning generated when making a prediction
		simplefilter("ignore") # ignore generated warnings
		return model.predict(X, return_std=True)

def opt_acquisition(X, y, model): # optimize the acquisition function
	samples = [ran(stacks) for prtcl in range(particles)] #generate initial input
	samples = samples.reshape(particles, len(ins[0])) #reshape inputs
	scores = acquisition(X, samples, model) # calculate the acquisition function for each sample
	ix = np.argmax(scores) # locate the index of the largest scores
	return samples[ix, 0]

def acquisition(X, Xsamples, model): # probability of improvement acquisition function
	yhat, _ = surrogate(model, X) # calculate the best surrogate score found so far
	best = max(yhat)
	mu, std = surrogate(model, Xsamples) # calculate mean and stdev via surrogate function
	probs = norm.cdf((mu - best) / (std+1E-9)) # calculate the probability of improvement
	return probs

if __name__=='__main__':
    loop=100 #number of times algorithm will itterate
    stacks=2
    particles=10

    pool=multiprocessing.Pool(particles) #create pool
    ins = [ran(stacks) for prtcl in range(particles)] #generate initial input
    rs = [pool.apply_async(runfile, args=(x)) for x in ins] #run LSTM code based on generated input values
    runs = [r.get() for r in rs] #get the result from processes

    inputs = ins.reshape(particles, len(ins[0])) #reshape inputs
    outputs = runs.reshape(particles, 1)

    model = GaussianProcessRegressor() # define the model
    model.fit(inputs, outputs) # fit the model

    for i in range(loop): #optimization algorithm
        next_input = opt_acquisition(ins, runs, model)
        print('next_input: ',next_input)
        out = runfile(next_input) #sample the point

        nin = next_input.reshape(1, len(next_input)) #reshape inputs to match the rest
        estimation = surrogate(model, [[nin]]) #generate estimation

        inputs = ((inputs, [[next_input]])) #add new point to inputs and outputs
        outputs = ((outputs, [[out]]))

        model.fit(inputs, outputs) # fit the model
        print('np.argmax(outputs): ',np.argmax(outputs))