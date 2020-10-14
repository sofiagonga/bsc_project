# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 22:51:10 2020

@author: sofia
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def coefficients(truncate_mode):
    """
    Creates an multi-dimensional array with the coefficients that will be assigned
    to the sines. The random choice has a normal bias.
    
    """
    coeffs = [[[np.random.normal(0, 1/(i+j+k)) for i in range(1,truncate_mode)] for j in range(1,truncate_mode)] for k in range(1,truncate_mode)]

    return coeffs

def random_field(coefficients, pos=[], random_phase = True):
    """
    For a given position and multi dimensional array of coefficients, it returns 
    the total gravitational (scalar) gravitational field for a given position.
    We can choose to have random phase for even more randomness.
    
    """
    field = 0 
    for i in range(1,len(coefficients)):
        for j in range(1, len(coefficients)):
            for k in range(1, len(coefficients)):
                if random_phase == True:
                    phase1= np.random.normal(0,(i+j+k)/10)
                    phase2= np.random.normal(0,(i+j+k)/10)
                    phase3= np.random.normal(0,(i+j+k)/10)
#                    phase1=random.choice(np.arange(-np.pi/2, np.pi/2, 0.1))
#                    phase2=random.choice(np.arange(-np.pi/2, np.pi/2, 0.1))
#                    phase3=random.choice(np.arange(-np.pi/2, np.pi/2, 0.1))
                else: 
                    phase1=0
                    phase2=0
                    phase3=0
                field += (coefficients[i][j][k]*np.sin(i*pos[0]+phase1)*np.sin(j*pos[1]+phase2)*np.sin(k*pos[2]+phase3))
    return field
    
    
coeffs = coefficients(25)
box_length = 10
values = np.arange(0,box_length,0.01)
list_test = []

for i in range(len(values)):
    list_test.append(random_field(coeffs,[1,1,values[i]]))
    
print(np.mean(list_test))
plt.plot(values, list_test)
plt.show()
