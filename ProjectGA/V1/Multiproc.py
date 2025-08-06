#GA - Reduced Forecast
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

def Evaluation(look, hid, re, lr, epoch, bat, dat): #define function to run LSTM code
    data=LSTMproject.arrange(look) #rearrange dataset
    ferr=LSTMproject.LSTMmodel(look, hid, re, lr, epoch, bat, dat, data) #generate LSTM and calculate error

    return ferr #return error

def Initialize(stacks): #generate random input values/Initialize population
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

def best(results, j): #find the index of the best value (not including index j)
    best = 999999
    n=0
    for i in range(results):
        if (results<best and i!=j):
            best=results
            n=i
    return i

def RanSelect(vector, val): #take random index
    while True:
        value=random.random() #generate random float between 0 and 1
        total=0
        for i in range(vector):
            if value>vector[i] and val!=i:
                return i

class parameters:
    def __init__(self, inputs, error):
        self.inputs = inputs
        self.error = error

if __name__=='__main__':
    loop=100 #number of times algorithm will itterate
    stacks=2 #LSTM stack size
    population=10 #population size
    crossover=0.6
    mutation=0.3

    pool = multiprocessing.Pool(population) #create pool
    pop = [Initialize(stacks) for prtcl in range(population)] #Initialize population

    for i in range(loop):
        rs = [pool.apply_async(Evaluation, args=(x)) for x in pop] #Evaluate solutions in parallel
        runs = [r.get() for r in rs] #Extract and store error from each solution

        #fitness
        fitness = runs.copy()
        totalerr = 0
        for j in range(population):
            if fitness[j] == 0:
                fitness[j]=0
            elif fitness[j] < 0:
                fitness[j] = -(1/runs[j])
            else:
                fitness[j] = (1/runs[j])
            totalerr =+ fitness[j]

        #probability
        probability = fitness.copy()
        for j in range(population):
            probability[j] = fitness[j]/totalerr

        #transfer best 2 results to new population
        new_pop = pop.copy() #store population to avoid overwriting changes
        new_runs = runs.copy()

        best1 = best(runs, population) #find index of best result
        best2 = best(runs, best1) #find index of 2nd best result
        new_pop[population-1] = pop[best1] #transfer best 2 results to next population
        new_pop[population-2] = pop[best2]

        #tournament selection
        for j in range(population-2): #dont include best 2 results
            candidate1 = RanSelect(probability, population) #index of candidate 1
            candidate2 = RanSelect(probability, candidate1) #index of candidate 2

            if runs[candidate1]<runs[candidate2]: #pick candidate 1 if it has better a better error
                new_pop[j] = pop[candidate1]
                new_runs[j] = runs[candidate1]
            else: #pick candidate 1 if it has better a better error
                new_pop[j] = pop[candidate2]
                new_runs[j] = runs[candidate2]

        pop = new_pop #overwrite population and error once changes are complete
        runs = new_runs

        #crossover
        new_pop = pop.copy()
        new_runs = runs.copy()

        for j in range(0, population-2, 2):
            parent1 = pop[j]
            parent2 = pop[j+1] 

            for k in range(len(parent1)): #for each parameter
                randval=random.random() #generate random number
                if randval<crossover: #check if random value is less than crossover probability
                    parent1[k] = pop[j+1][k]
                    parent2[k] = pop[j][k]

            new_pop[j] = parent1 #store results
            new_pop[j+1] = parent2

        pop = new_pop #transfer results to population 

        #mutation
        for j in range(population-2): #random mutation
            mutant = Initialize(stacks) #create random set of parameters
            for k in range(len(pop[j])):
                value=random.random() #generate random float between 0 and 1
                if value<mutation: #check if random value is below mutation rate
                    pop[j][k]=mutant[k] #store mutation