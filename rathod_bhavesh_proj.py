# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:56:01 2020

@author: bhavrathod
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import scipy

# Inputs: 4 csv files: dac details, policy details, wind data and solar data
# Outputs: DAC performance csv file and incentive benefits csv file in folder this code resides in
# Purpose: main function
def main():
    
    windData,solData,sizeToSim = importCurtData()
    scaleW,scaleS = fitCurveToData(windData, solData)
    # Power in MW below
    windSim,solSim = runSimulations(scaleW,scaleS,sizeToSim)
    dacData = importDacData()
    dacData = detDacOutput(dacData,windSim,solSim,sizeToSim)
    incentives = readPolicyScenarios()
    assessPolicyBenefits(incentives,dacData)

# Inputs: none
# Outputs: pandas DF with DAC electricity requirements and heat requirements per tCO2 captured
# Purpose: define the characteristics 
def importDacData():
    dacData = pd.read_csv('bhavesh_rathod_DacDetails.csv')
    #print(dacData)
    return dacData

# Inputs: none
# Outputs: pandas DF with curtailed capacity in kW and timeframe of observations (5 minute window)
# Purpose: read in electricity curtailment data from CAISO
def importCurtData():
    
    # Data is in 5 minute resolutions, so for 1 year, we would need 365*24*12 = 105120 time iterations
    sizeToSim = 365*24*12
    #print(sizeToSim)
    
    windData = np.loadtxt('wind.csv', dtype=int, skiprows=1)
    solData = np.loadtxt('solar.csv', dtype=int, skiprows=1)
    #print(len(windData),len(solData))
    return windData,solData,sizeToSim

# Other fitting attempts
# Take the following function because https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html says so
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3667.htm
# def func(x, beta, mu):
#     return 1/beta*np.exp(-(x-mu)/beta)
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# Inputs: solar and wind curtailed electricity data
# Output: tuple with scale of exponential distributions to use
# Purpose: return the values that can be used as inputs to np.random.exponential to sample curtailed electricity
def fitCurveToData(windData,solData):
    
    # Plotted data for reference
    wFreqCount = np.bincount(windData)
    # Cleansing data, first value is 0,0 and can mess with the fitting
    wFreqCount = np.delete(wFreqCount,0)
    xMax = len(wFreqCount)
    xValWind = range(0,xMax)
    xValWind = np.asarray(xValWind)
    # Wind fitting
    # Guidance from https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
    dist_name = 'expon'
    dist = getattr(scipy.stats, dist_name)
    locW, scaleW = dist.fit(wFreqCount)
    label = "Fitted location = "+str(locW)+", scale="+str(round(scaleW,2))
    # param is a tuple of floats, 
    # print(locW, scaleW)
    pdf_fitted = dist.pdf(xValWind, loc=locW, scale=scaleW)
    cdf_fitted = dist.cdf(xValWind, loc=locW, scale=scaleW)
    plt.plot(xValWind,wFreqCount)
    plt.xlabel('Curtailed wind electricity capacity (MW)')
    plt.ylabel('Frequency')
    plt.text(200,400,label)
    plt.savefig('curtailedWindCAISO.jpg')
    plt.clf()
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlabel('Curtailed electricity capacity (MW)')
    plt.ylabel('Probability distribution (fitted)')
    plt.savefig('pdf_curtailedWindCAISO.jpg')
    plt.clf()
    plt.plot(cdf_fitted, 'g', label=dist_name)
    plt.xlabel('Curtailed electricity capacity (MW)')
    plt.ylabel('Cumulative probability (fitted)')
    plt.savefig('cdf_curtailedWindCAISO.jpg')
    plt.clf()
    
    sFreqCount = np.bincount(solData)
    # Cleansing data
    sFreqCount = np.delete(sFreqCount,0)
    xMax = len(sFreqCount)
    xValSol = range(0,xMax)
    xValSol = np.asarray(xValSol)
    # solar fitting
    # Guidance from https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
    dist_name = 'expon'
    dist = getattr(scipy.stats, dist_name)
    locS, scaleS = dist.fit(sFreqCount)
    # param is a tuple of floats, 
    # print(locS, scaleS)
    pdf_fitted = dist.pdf(xValSol, loc=locS, scale=scaleS)
    cdf_fitted = dist.cdf(xValSol, loc=locS, scale=scaleS)
    label = "Fitted location = "+str(locS)+", scale="+str(round(scaleS,2))
    plt.plot(xValSol,sFreqCount)
    plt.xlabel('Curtailed solar electricity capacity (MW)')
    plt.ylabel('Frequency')
    plt.text(2000,600,label)
    plt.savefig('curtailedSolarCAISO.jpg')
    plt.clf()
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlabel('Curtailed electricity capacity (MW)')
    plt.ylabel('Probability distribution (fitted)')
    plt.savefig('pdf_curtailedSolarCAISO.jpg')
    plt.clf()
    plt.plot(cdf_fitted, 'g', label=dist_name)
    plt.xlabel('Curtailed electricity capacity (MW)')
    plt.ylabel('Cumulative probability (fitted)')
    plt.savefig('cdf_curtailedSolarCAISO.jpg')
    plt.clf()
    
    return scaleW,scaleS

# Inputs: # of times to run simulations, scale, time periods sizeToSim, 1 sample for each time period
# Outputs: np array with simulation outcomes
# Purpose: run simulations and return sampled values
def runSimulations(optValW,optValS,sizeToSim):
    
    windSim = np.random.exponential(scale=optValW, size=sizeToSim)
    solSim = np.random.exponential(scale=optValS, size=sizeToSim)
    #print(windSim)
    #print(solSim)
    return windSim,solSim

# Inputs: simulation output for curtailed electricity, DAC performance data
# Output: timeframes when DAC plant can operate, policy gain
# Purpose: to determine the time for which DAC plant can operate using given curtailed electricty distribution
def detDacOutput(dacData,windSim,solSim,sizeToSim):
    # Covert each power value in sim data (MW) to energy over 5 minute time period in kWh
    windEnergy = windSim*(5/60)*1000
    # Covert each power value in sim data (MW) to energy over 5 minute time period in kWh
    solEnergy = solSim*(5/60)*1000
    
    # sum of 5 minute intervals for which DAC is operational
    dacOperational = 0
    dacBrands = len(dacData)
    
    # Add columns to data
    dacData['windValidTimes'] = [0,0,0,0]
    dacData['windTonsCo2Captd'] = [0,0,0,0]
    dacData['windTotWorkingHrs'] = [0,0,0,0]
    #print(dacData.shape[1])
    
    for dac in range(dacBrands):
        for t in range(sizeToSim):
            if windEnergy[t] > dacData.iloc[dac,2]:
                dacOperational += 1
        dacData.iloc[dac,3] = dacOperational
        dacData.iloc[dac,4] = dacOperational
        dacData.iloc[dac,5] = dacOperational*5/60
        dacOperational = 0
    
    dacData['solValidTimes'] = [0,0,0,0]
    dacData['solTonsCo2Captd'] = [0,0,0,0]
    dacData['solTotWorkingHrs'] = [0,0,0,0]
    #print(dacData.shape[1])
    
    for dac in range(dacBrands):
        for t in range(sizeToSim):
            if solEnergy[t] > dacData.iloc[dac,2]:
                dacOperational += 1
        dacData.iloc[dac,6] = dacOperational
        dacData.iloc[dac,7] = dacOperational
        dacData.iloc[dac,8] = dacOperational*5/60
        dacOperational = 0
    dacData.to_csv('dacPerformanceDetails.csv')
    return dacData

# https://www.energy.gov/sites/prod/files/2019/10/f67/Internal%20Revenue%20Code%20Tax%20Fact%20Sheet.pdf
def readPolicyScenarios():
    
    # read in low, medium, high price incentives for carbon capture where utility of carbon can influcence the prices.
    
    incentivesData = pd.read_csv('bhavesh_rathod_policyDetails.csv')
    return incentivesData

# Input: tons of CO2 captured with DAC using curtailed electricity
# Output: monetary benefit over 1 year of operations
# Determine the monetary benefit over 1 year of operations under various policies
def assessPolicyBenefits(incentives,dacData):
        
    # for each incentive, calculate profits for leading brands
    for case in range(incentives.shape[0]):
        # Add a column for incentive description
        dacData[incentives.iloc[case,0]] = [0,0,0,0]
        profit = incentives.iloc[case,1]
        #print(dacData)
        #print("new loops",dacData.iloc[0,-1])
        for dac in range(dacData.shape[0]):
            # For each dac, calculate profits
            dacData.iloc[dac,-1] = profit*dacData.iloc[dac,-6-case]
            #print(dac,dacData.iloc[dac,-1],dacData.iloc[dac,-6-case])
    dacData.to_csv('policyOutcomeDetails_wind.csv')

    # for each incentive, calculate profits for leading brands
    for case in range(incentives.shape[0]):
        # Add a column for incentive description
        dacData[incentives.iloc[case,0]] = [0,0,0,0]
        profit = incentives.iloc[case,1]
        #print(dacData)
        #print("new loops",dacData.iloc[0,-1])
        for dac in range(dacData.shape[0]):
            # For each dac, calculate profits
            dacData.iloc[dac,-1] = profit*dacData.iloc[dac,-3-case]
            #print(dac,dacData.iloc[dac,-1],dacData.iloc[dac,-3-case])
    dacData.to_csv('policyOutcomeDetails_solar.csv')

main()