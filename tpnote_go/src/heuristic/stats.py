##############################################################################
#                   TP Go - IA - S8 - I2G1 - ddelpy - allaguierce
#                           -*- coding: utf-8 -*-
#                             Stats script file.
##############################################################################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import gzip, os.path, json, urllib.request

NB_COLUMNS = NB_LINES = 8

# From `src/go.py`.
# <>
len_data: int = 41553
N: int = int(len_data / 3)
# </>

# !!! ---------------------- !!!
# Notebook wants this:
# <>
#%matplotlib inline
# </>
# !!! ---------------------- !!!

##############################################################################
# To find the number of right guesses.
##############################################################################

def find_number_right_guesses(file_predictions: str, yield_abs: float = 0.05) -> float:
    results: list = []
    with open(file_predictions, "r") as f:
        for line in f.readlines():
            formatted: list = line.replace(';', '').split(' ')
            guess: float = float(formatted[1])
            answer: float = float(formatted[-1])

            if abs(guess - answer) < yield_abs:
                results.append(1) # Means right guess.
            else:
                results.append(0) # Means the guess was wrong.

    if results == []:
        return 0
    return round(sum(results) / len(results), 2)

yields: list = [ 0.05, 0.10, 0.20, 0.30, 0.40, 0.50 ]
for y in yields:
    print(f'\nFor {N} training inputs over {len_data} inputs, \n \
          the percentage of right guesses, with a yield of {y * 100}%, \n \
          on the testing inputs is: \
          {find_number_right_guesses("./my_predictions.txt", y)} \n \
    ')

##############################################################################
# Display training curves.
##############################################################################

#model: torch.Model = torch.load('./model.pt')

##############################################################################
# EOF
##############################################################################

