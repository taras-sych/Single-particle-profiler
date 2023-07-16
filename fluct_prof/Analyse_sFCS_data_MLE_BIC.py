# -*- coding: utf-8 -*-
#
#
# Analyse data for logN or dlogN behaviour
# Based on the Matlab code used in:
# Schneider, F. et al. Statistical Analysis of Scanning Fluorescence Correlation Spectroscopy Data Differentiates Free from Hindered Diffusion. ACS Nano 12, 8540–8546 (2018).    
#
# Lognormal (logN) behaviour indicates free diffusion
# Double lognormal (dlogN) behaviour indicates a second component usually attributed to apopulation of molecules undergoing trapping (transient halts in the diffusion path)
#
# This script is a cleaned up version of "analyse_trapping_MLE_BIC"
# Has not been tested as extensively as the Matlab code. 
# Annotated and commented for Jan Schlegel
#
# Falk Schneider @faldalf 02/10/2021
#
# Questions and feedback are welcome
# No warranties
#%% Improts
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
from tkinter import filedialog
from tkinter import *
import os
from fnmatch import fnmatch

#%% Define Functions

#def load_sFCS_data (): 
## Small reader function to get sFCS data into Python
## Supports data right out of FoCuS_scan or from Falk's simulation pipelines
## outputs the data as a pandas dataframe

#    root = Tk()
#    root.filename =  filedialog.askopenfilename(title = "File to analyse",filetypes = (("FCS Fittingparameters",".csv .xlsx"),("all files","*.*")))
#    filename = root.filename
#    filename = filename.replace("/","\\\\")
#    root.withdraw()

#    if 'xlsx' in filename:
#        data = pd.read_excel (filename)
#        data = data.drop(data.index[-1]) # removes the last enrty (end from FoCuS_scan)
#        print (os.path.basename (filename), ' has been loaded')
#        print ('Warning: Last row has been removed')
#    elif 'csv' in filename:
#        data = pd.read_csv (filename)
#    else:
#        print ('Error: Can\'t read file format. Please provide .xlsx or .csv')
        
#    return data, filename


def NLogLGauss(theta,data):
# NLL_Gauss retruns the Negative logarithmic Likelihood value for Gaussian
# distributed data using theta1 for mu and theta2 for sigma. 

    mu = theta[0]
    sig = theta[1]
    
    l = -0.5*np.log(2*np.pi*sig**2)-(data-mu)**2/(2*sig**2)
    # Validated here: http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
    NLL_Gauss = - np.sum(l)
    return NLL_Gauss


def NLogLLognorm(theta,data):
# NLL_Gauss retruns the Negative logarithmic Likelihood value for Gaussian
# distributed data using theta1 for mu and theta2 for sigma. 
 
    mu = theta[0]
    sig = theta[1]

    l = -0.5*np.log(2*np.pi*sig**2)-np.log(data)-((np.log(data))**2)/(2*sig**2)+(np.log(data)*mu)/(sig**2)-(mu**2)/(2*sig**2)

    NLL_Lognorm = - np.sum(l);

    return NLL_Lognorm

def NLogLDoubleLognorm (theta,data):
# % NLL_Gauss retruns the Negative logarithmic Likelihood value for Gaussian
# % distributed data using theta1 for mu and theta2 for sigma. 
    
    mu1 = theta[0]
    sigma1 = theta[1]
    mu2 = theta[2] 
    sigma2 = theta[3] 
    B = theta[4]
    #B2 = theta[4]


    l = - 0.5*np.log(2*np.pi) - np.log (data) + np.log( B*sigma1**(-1) * np.exp(-(np.log(data)-mu1)**2/(2*sigma1**2))+ ((1-B)*sigma2**(-1) * np.exp(-(np.log(data)-mu2)**2/(2*sigma2**2)))) 
    
    NLL_DoubleLognorm = -np.sum(l)
    
    return NLL_DoubleLognorm 


def model_selection_RL (data, initial_guess = [4,1], plot = 'on'):
    # Calculates the relative likelihood based on the functions above for data being normally or lognormally distributed
    # data: Should be a 1D vector containing the data to be tested
    # initial_guess is a 1D vector containing 2 values as initial guesses for mu and sigma. 
    # bounds are provided hard-coded here ... that is not optimal but really helps with convergence. Probably need to be varied for special cases
    # 
    
    
    # Here comes the MLE for Gauss
    bnds = ((0.1, 100), (0.1, 100))
    optimised_NLogLikelihood_Gauss = opt.minimize (NLogLGauss, initial_guess, data, bounds = bnds)
    mu_MLE_Gauss = optimised_NLogLikelihood_Gauss.x[0]
    sigma_MLE_Gauss = optimised_NLogLikelihood_Gauss.x[1]
    
    # Here comes the analogous MLE for lognorm
    bnds = ((0.1, 10), (0.1, 10))
    optimised_NLogLikelihood_logN = opt.minimize (NLogLLognorm, initial_guess, data, bounds = bnds)
    mu_MLE_logN = optimised_NLogLikelihood_logN.x[0]
    sigma_MLE_logN = optimised_NLogLikelihood_logN.x[1]
    
    # Here comes the new bit for the double logN
    # Initialisation is a bit dirty ... can't come up with a better solution at the moment
    initial_guess_dlogN = [initial_guess[0], initial_guess [1], initial_guess[0], initial_guess[1], 0.5]
    bnds = ((0.1, 10), (0.1, 10), (0.1, 3), (0.1, 3), (0.05, 0.95))

    optimised_NLogLikelihood_dlogN = opt.minimize (NLogLDoubleLognorm, initial_guess_dlogN, data, bounds = bnds)
    mu_MLE_dlogN = optimised_NLogLikelihood_dlogN.x[0]
    sigma_MLE_dlogN = optimised_NLogLikelihood_dlogN.x[1]
    mu2_MLE_dlogN = optimised_NLogLikelihood_dlogN.x[2]
    sigma2_MLE_dlogN = optimised_NLogLikelihood_dlogN.x[3]
    B_MLE_dlogN = optimised_NLogLikelihood_dlogN.x[4]
    


    # Calculate the BIC Values from MLE according to:
    # https://uk.mathworks.com/help/econ/aicbic.html
    # We assume two parameters for Gauss and LogN (mu and sigma)    
    logL_Gauss = - optimised_NLogLikelihood_Gauss.fun
    bic_Gauss = -2*(logL_Gauss) + 2 * np.log (len(data)) 
    logL_logN =  - optimised_NLogLikelihood_logN.fun
    bic_logN = -2*(logL_logN) + 2 * np.log (len(data))
    
    #dlogN
    logL_dlogN = - optimised_NLogLikelihood_dlogN.fun
    bic_dlogN = -2*(logL_dlogN) + 5 * np.log (len(data)) # We have 5 variables now

    #Calculate Relative Likelihood values (pesudo p-values for how much better a model represents the data compared to another model based on BIC)
    RL_Gauss = np.exp ((np.min([bic_Gauss, bic_logN, bic_dlogN])-bic_Gauss)/2)
    RL_logN = np.exp ((np.min([bic_Gauss, bic_logN, bic_dlogN])-bic_logN)/2)
    RL_dlogN = np.exp ((np.min([bic_Gauss, bic_logN, bic_dlogN])-bic_dlogN)/2)
    
    # Here comes some plotting
    if plot == 'on':
        fig, axes = plt.subplots(nrows=1, ncols=2,  figsize=(10, 6))
        ax0, ax1 = axes.flatten()
        ax0.hist (data, edgecolor='black', linewidth=1, density = True, label = 'Data')
        plt.title ('Gaussian Random data')
        x_max = np.max (data)
        x_min = np.min (data)
        x_MLE = np.linspace(x_min,x_max,100)
        y_MLE_Gauss =  1/(sigma_MLE_Gauss * np.sqrt(2 * np.pi))*np.exp( - (x_MLE - mu_MLE_Gauss)**2 / (2 * sigma_MLE_Gauss**2) )
        y_MLE_logN = 1/(x_MLE*sigma_MLE_logN*np.sqrt(2*np.pi)) * np.exp (- (np.log(x_MLE)-mu_MLE_logN)**2/(2*sigma_MLE_logN**2))
        
        y_MLE_dlogN = B_MLE_dlogN * (1/(x_MLE*sigma_MLE_dlogN*np.sqrt(2*np.pi)) * np.exp (- (np.log(x_MLE)-mu_MLE_dlogN)**2/(2*sigma_MLE_dlogN**2))) + (1-B_MLE_dlogN) * (1/(x_MLE*sigma2_MLE_dlogN*np.sqrt(2*np.pi)) * np.exp (- (np.log(x_MLE)-mu2_MLE_dlogN)**2/(2*sigma2_MLE_dlogN**2)))
        ax0.plot (x_MLE, y_MLE_Gauss, 'r', label = 'MLE (Gauss)')
        ax0.plot (x_MLE, y_MLE_logN, 'g', label = 'MLE (Lognorm)')
        ax0.plot (x_MLE,y_MLE_dlogN, '--y', label = 'MLE (Double Lognorm)')
        ax0.legend ()
        ax0.set_xlabel ('Data_in')
        ax0.set_ylabel ('Probability (a.u.)')
    
        x = np.array([1,2,3])
        my_xticks = ['Gauss','LogN','dLogN']
        ax1.set_xticks (x)
        ax1.set_xticklabels (my_xticks)
        ax1.set_xlim ([0,4])
        ax1.plot (x, [RL_Gauss, RL_logN, RL_dlogN], '-bo')
        ax1.set_title ('Model Selection')
        ax1.set_ylabel ('Relative Likelihood')
        ax1.set_ylim ([-0.1, 1.1])

        plt.tight_layout()
        plt.show()

    # Let's save the interesting data as well as the figure
    
    RL_results = {}
    RL_results ['RL_Gauss'] = RL_Gauss
    RL_results ['RL_LogN'] = RL_logN
    RL_results ['Gauss_mu_sigma'] = [mu_MLE_Gauss, sigma_MLE_Gauss]
    RL_results ['LogN_mu_sigma'] = [mu_MLE_logN, sigma_MLE_logN]
    RL_results ['RL_dLogN'] = RL_dlogN
    RL_results ['dLogN_fitting'] = [mu_MLE_dlogN, sigma_MLE_dlogN, mu2_MLE_dlogN, sigma2_MLE_dlogN, B_MLE_dlogN]
    #print ('Model Selection Done')
    
    return RL_results, fig

##%% Run script

## User defined input:
#cutoff = np.array ([0, 500]) # Upper and lower bounds for transit times. Do not consider too small or too big transit times

## Load data
#data, filename = load_sFCS_data ()
## Process transit time data
#data = data [data['txy1']< cutoff[1]]
#data = data [data['txy1']> cutoff[0]]

#results, fig = model_selection_RL (data['txy1'], initial_guess = [4,1], plot = 'on')

#filepath = os.path.dirname (filename)
#filepath = filepath +'\\\\'
#fig_path = filepath+'BIC.png'   
#fig.savefig(fig_path, bbox_inches='tight')

#print ("\n", "Relative Likelihood (free diffusion)", results['RL_LogN'], "\n", "Relative Likelihood (hindered diffusion) ", results['RL_dLogN']) # Let's print out the RL values. 











