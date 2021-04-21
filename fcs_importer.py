#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:39:58 2020

@author: tarassych
"""
#--------------------------------------------------- 

"""
Classes with data types
#--------------------------------------------------- 
"""

import numpy as np

class XY_plot:
    def __init__ (self, x_arg, y_arg):
        self.x = x_arg
        self.y = y_arg

class fcs_channel:
    
    def __init__ (self, name_arg, fluct_arr_arg, auto_corr_arr_arg,photon_count_arr_arg, pulse_distance_arr_arg):
        self.name = name_arg
        self.fluct_arr = fluct_arr_arg
        self.auto_corr_arr = auto_corr_arr_arg
        self.photon_count_arr = photon_count_arr_arg
        self.pulse_distance_arr = pulse_distance_arr_arg

class fcs_cross:
    
    def __init__(self, cross_corr_arr_arg, description_arg):
        self.cross_corr_arr = cross_corr_arr_arg
        self.description = description_arg
    

class Dataset_fcs:
    
    def __init__ (self,channels_number_arg, channels_list_arg, cross_list_arg):
        self.channels_number = channels_number_arg
        self.channels_list = channels_list_arg
        self.cross_list = cross_list_arg

class Full_dataset_fcs:
    
    def __init__ (self, repetitions_arg, dataset_list_arg):
        self.repetitions = repetitions_arg
        self.datasets_list = dataset_list_arg
        
    
        
#---------------------------------------------------  
        
#---------------------------------------------------
        
"""
Functions for .fcs
#--------------------------------------------------- 
"""

def Find_repetitions (list_file):
    repetitions = 0

    for i in range(0,len(list_file)):
        if list_file[i].__contains__("Repetition = ") :
        
            str1 , str2 = list_file[i].split(' = ')
            repetitions_temp = int(str2)
        
            if repetitions_temp > repetitions:
                repetitions = repetitions_temp
            
    return repetitions+1


def Fill_datasets_fcs (list_file, repetitions):
    
    #----find number of channels--------------
    
    counter = 0
    for j in range(0,len(list_file)):
            if list_file[j].__contains__("Repetition = " + str(0)) :
                counter+=1
                
    channels_number = int(np.sqrt(counter))
    #-----------------------------------------
    
    list_of_repetions = []
    
    i=0;
    while i<repetitions:
        
        #----save positions of lines "Repetition = i"
        rep_list = []
        for j in range(0,len(list_file)):
            if list_file[j].__contains__("Repetition = " + str(i)) :
                rep_list.append(j)
        
        channels_list = []
        
        # -------read channels---------------------------------------
        
        for j in range (0,channels_number):
            index = rep_list[j]
            str1 , str2  = list_file[index+1].split(' = ')
            channel_name = str2
            
            k_start = index
            
            flag = False
            k = index
            while flag == False:
                k+=1
                if list_file[k].__contains__("CorrelationArraySize") :
                    flag = True
                
            
            
            fluct = list_file[k_start+7 : k-1]
            n = len(fluct)

            x = np.empty([n], dtype=float)
            y = np.empty([n], dtype=float)

            for ii in range(0,n):
                fluct[ii] = fluct[ii].strip('\n')
                fluct[ii] = fluct[ii].split()
                x[ii] = float(fluct[ii][0])
                y[ii] = float(fluct[ii][1])
            
            Fluct_arr = XY_plot(x,y)
            
            k_start = k
            
            flag = False
            
            while flag == False:
                k+=1
                if list_file[k].__contains__("PhotonCountHistogramArraySize") :
                    flag = True
                
            fluct = list_file[k_start+2 : k-1]
            n = len(fluct)

            x = np.empty([n], dtype=float)
            y = np.empty([n], dtype=float)

            for ii in range(0,n):
                fluct[ii] = fluct[ii].strip('\n')
                fluct[ii] = fluct[ii].split()
                x[ii] = float(fluct[ii][0])
                y[ii] = float(fluct[ii][1])
                
            Corr_arr = XY_plot(x,y)
            
            k_start = k
            
            flag = False
            
            while flag == False:
                k+=1
                if list_file[k].__contains__("PulseDistanceHistogramArraySize") :
                    flag = True
                
            fluct = list_file[k_start+2 : k-1]
            n = len(fluct)

            x = np.empty([n], dtype=float)
            y = np.empty([n], dtype=float)

            for ii in range(0,n):
                fluct[ii] = fluct[ii].strip('\n')
                fluct[ii] = fluct[ii].split()
                x[ii] = float(fluct[ii][0])
                y[ii] = float(fluct[ii][1])
                
            Photon_arr = XY_plot(x,y)
            
            k_start = k
            
            flag = False
            
            while flag == False:
                k+=1
                if list_file[k].__contains__("CountRateCutRegionArraySize") :
                    flag = True
                
            fluct = list_file[k_start+2 : k-1]
            n = len(fluct)

            x = np.empty([n], dtype=float)
            y = np.empty([n], dtype=float)

            for ii in range(0,n):
                fluct[ii] = fluct[ii].strip('\n')
                fluct[ii] = fluct[ii].split()
                x[ii] = float(fluct[ii][0])
                y[ii] = float(fluct[ii][1])
                
            Pulse_arr = XY_plot(x,y)
            
            fluct_channel = fcs_channel(channel_name, Fluct_arr, Corr_arr, Photon_arr, Pulse_arr)
            
            channels_list.append(fluct_channel) 
            
        # --------------------------------------------------------------
        
        cross_list = []
        # -------read cross-correlation---------------------------------
        for j in range (channels_number,channels_number * channels_number):
            index = rep_list[j]
            str1 , str2  = list_file[index+1].split(' = ')
            channel_name = str2
            
            k_start = index
            
            flag = False
            k = index
            while flag == False:
                k+=1
                if list_file[k].__contains__("PhotonCountHistogramArraySize") :
                    flag = True
                
            
            
            fluct = list_file[k_start+9 : k-1]
            n = len(fluct)

            x = np.empty([n], dtype=float)
            y = np.empty([n], dtype=float)

            for ii in range(0,n):
                fluct[ii] = fluct[ii].strip('\n')
                fluct[ii] = fluct[ii].split()
                x[ii] = float(fluct[ii][0])
                y[ii] = float(fluct[ii][1])
            
            Corr_arr = XY_plot(x,y)
            
            corr_channel = fcs_cross(Corr_arr, channel_name)
            
            cross_list.append(corr_channel)
        #---------------------------------------------------------------
        
        
        channels_list = Correct_channels (channels_list)
            
        
        dataset = Dataset_fcs(channels_number, channels_list, cross_list)
        
        list_of_repetions.append(dataset)
        i+=1
    
    return Full_dataset_fcs(repetitions, list_of_repetions)


def Correct_channels(channels_list):
    
    
    x1 = channels_list[0].fluct_arr.x
    y1 = channels_list[0].fluct_arr.y
    x2 = channels_list[1].fluct_arr.x
    y2 = channels_list[1].fluct_arr.y
    
    if len(y1) > len(y2):
        nn = len(y1) - len(y2)
        print(nn)
        for inin in range (0,nn):
            y1 = np.delete(y1,len(y1)-1)
            x1 = np.delete(x1,len(y1)-1)
        
        
        
    if len(y2) > len(y1):
        nn = len(y2) - len(y1)
        for inin in range (0,nn):
            y2 = np.delete(y2,len(y2)-1)
            x2 = np.delete(x2,len(y1)-1)
            
    channels_list[0].fluct_arr.x = x1
    channels_list[0].fluct_arr.y = y1
    channels_list[1].fluct_arr.x = x2
    channels_list[1].fluct_arr.y = y2
    
    return channels_list
"""
Functions for .SIN
#--------------------------------------------------- 
"""

def Fill_datasets_sin (list_file):
    
    flag = False
    k = 0
    while flag == False:
        k+=1
        if list_file[k].__contains__("CorrelationFunction") :
            flag = True
            
    k_start = k    
            
    flag = False
    while flag == False:
        k+=1
        if list_file[k].__contains__("RawCorrelationFunction") :
            flag = True
            
    fluct = list_file[k_start+1 : k-2]
    n = len(fluct)

    delay = np.empty([n], dtype=float)
    acorr_1 = np.empty([n], dtype=float)
    acorr_2 = np.empty([n], dtype=float)
    cross_corr_12 = np.empty([n], dtype=float)
    cross_corr_21 = np.empty([n], dtype=float)

    for ii in range(0,n):
        fluct[ii] = fluct[ii].strip('\n')
        fluct[ii] = fluct[ii].split()
        delay[ii] = float(fluct[ii][0])
        acorr_1[ii] = float(fluct[ii][1])
        acorr_2[ii] = float(fluct[ii][2])
        cross_corr_12[ii] = float(fluct[ii][3])
        cross_corr_21[ii] = float(fluct[ii][4])
            
            
            
            
    flag = False
    while flag == False:
        k+=1
        if list_file[k].__contains__("IntensityHistory") :
            flag = True
            
    k_start = k
            
    flag = False
            
    while flag == False:
        k+=1
        if list_file[k].__contains__("Histogram") :
            flag = True
                
    fluct = list_file[k_start+2 : k-2]
    n = len(fluct)

    x = np.empty([n], dtype=float)
    y1 = np.empty([n], dtype=float)
    y2 = np.empty([n], dtype=float)

    for ii in range(0,n):
        fluct[ii] = fluct[ii].strip('\n')
        fluct[ii] = fluct[ii].split()
        x[ii] = float(fluct[ii][0])
        y1[ii] = float(fluct[ii][1])
        y2[ii] = float(fluct[ii][2])
                
    Fluct_arr_1 = XY_plot(x,y1)
            
    Fluct_arr_2 = XY_plot(x,y2)
            
    Corr_arr_1 = XY_plot(delay,acorr_1)
            
    Corr_arr_2 = XY_plot(delay,acorr_2)
            
    Corr_arr_3 = XY_plot(delay,cross_corr_12)
            
    Corr_arr_4 = XY_plot(delay,cross_corr_21)
            
            
                
            
            
            
            
    fluct_channel_1 = fcs_channel("channel 1", Fluct_arr_1, Corr_arr_1, "no data", "no data")
            
    fluct_channel_2 = fcs_channel("channel 2", Fluct_arr_2, Corr_arr_2, "no data", "no data")
            
            
    channels_list = []
    channels_list.append(fluct_channel_1) 
    channels_list.append(fluct_channel_2)
            
    corr_channel_1 = fcs_cross(Corr_arr_3, "Cross correlation 1 - 2")
    corr_channel_2 = fcs_cross(Corr_arr_4, "Cross correlation 2 - 1")
            
    cross_list = []
    cross_list.append(corr_channel_1)
    cross_list.append(corr_channel_2)
    
    channels_list = Correct_channels (channels_list)
 
    dataset = Dataset_fcs(2, channels_list, cross_list)
        
    list_of_repetions = []
    
    list_of_repetions.append(dataset)
    
    
    return Full_dataset_fcs(1, list_of_repetions)