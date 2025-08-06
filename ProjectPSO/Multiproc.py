#PSO - Reduced Forecast
import multiprocessing
import LSTMproject

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

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

def compare(best, inputs, error): #compare input to local best
    if (int(best.error)>int(error)):
        best.positions=inputs
        best.error=error

def comp(globalval, localval): #compare local and global value
    if (int(globalval.error)>int(localval.error)):
        globalval.inputs=localval.inputs
        globalval.error=localval.error

def update(inputs, velocity, globalval, localval): #update inputs for next iteration
    globe=globalval.inputs #global parameters
    local=localval.inputs #local parameters
    inputs_list = list(inputs) # Convert tuple to list
    
    w = 0.6 #inertia (0.5 to 1)
    i_s = 0.5 #social influence (0 to 1)
    i_p = 0.5 #personal influence (0 to 1)
    a = 2 #acceleration coefficient (1.5 to 2)

    for i in range(len(inputs_list)):
        velocity[i] = w*velocity[i] + a*i_s*(local[i] - inputs_list[i]) + a*i_p*(globe[i] - inputs_list[i]) #velocity update
        inputs_list[i] = inputs_list[i] + velocity[i] #position update
        inputs_list[i]=int(inputs_list[i])

    return tuple(inputs_list), velocity #Convert list back to tuple

class parameters:
    def __init__(self, inputs, error):
        self.inputs = inputs
        self.error = error

if __name__=='__main__':
    loop=100 #number of times algorithm will itterate
    stacks=2 #number of LSTM stacks
    particles=10 #population size
    initial_vel = [0]*particles

    pool=multiprocessing.Pool(particles) #create pool

    ins = [ran(stacks) for prtcl in range(particles)] #initialize population
    vel = [initial_vel for prtcl in range(particles)] #initialize velocities at zero

    loc = [parameters(ins, 999999) for l in range(particles)] #local maximums
    glob=parameters((20, 27, 2, 591, 100, 21, 0.85), 5168) #global maximum

    for i in range(loop): #start algorithm
        rs = [pool.apply_async(runfile, args=(x)) for x in ins] #run LSTM code based on generated input values
        runs = [r.get() for r in rs] #get the result from processes

        for idx, r in enumerate(runs): 
            compare(loc[idx], ins[idx], r) #update local variables
            comp(glob, loc[idx]) #update global variable

        for idy, ru in enumerate(runs):
            ins[idy], vel[idy]=update(ins[idy], vel[idy], glob, loc[idy]) #modify parameters based on global and local values