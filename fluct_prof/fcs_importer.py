import numpy as np

import tkinter as tk
from tkinter import ttk

import copy

import pandas as pd

from fluct_prof import Correlation as corr_py





class XY_plot:
    def __init__ (self, x_arg, y_arg):
        self.x = x_arg
        self.y = y_arg

class fcs_channel:
    
    def __init__ (self, name_arg, fluct_arr_arg, auto_corr_arr_arg, short_name_arg):
        self.name = name_arg
        self.fluct_arr = fluct_arr_arg
        self.auto_corr_arr = auto_corr_arr_arg
        self.short_name = short_name_arg

        

        cr_list = []


        ys = 0
        counter = 1

        for i in range(len(self.fluct_arr.x)):
        
            ys += self.fluct_arr.y[i]

            if self.fluct_arr.x[i] >= counter:
                cr_list.append(ys)
                ys = 0
                counter +=1

        self.count_rate = np.mean(cr_list)/1000




class fcs_cross:
    
    def __init__(self, name_arg,  cross_corr_arr_arg, short_name_arg):
        self.name = name_arg
        self.cross_corr_arr = cross_corr_arr_arg
        self.short_name = short_name_arg


    

class Dataset_fcs:
    
    def __init__ (self,channels_number_arg, cross_number_arg, channels_list_arg, cross_list_arg):
        self.channels_number = channels_number_arg
        self.cross_number = cross_number_arg
        self.channels_list = channels_list_arg
        self.cross_list = cross_list_arg

class Full_dataset_fcs:
    
    def __init__ (self, repetitions_arg, dataset_list_arg):



        self.position = ''

        self.metadata = None

        self.blue_setup = None
        self.red_setup = None


        self.repetitions = repetitions_arg
        self.datasets_list = dataset_list_arg
        

        self.threshold_list = [None] * self.datasets_list[0].channels_number

        self.detection_how = None
        
        self.binning = 1
        self.peaks = {}
        self.peak_prominences = {}
        self.peak_widths = {}
        self.gp_fitting = [None] * repetitions_arg
        self.diff_fitting = {}
        self.N = {}
        self.cpm = {}
        self.diff_coeffs = {}

        self.export_dataframe = None

        for i in range(self.datasets_list[0].channels_number + self.datasets_list[0].cross_number):
            for j in range(repetitions_arg):
                self.diff_fitting[j, i] = None
                self.diff_coeffs[j, i] = None



        for i in range(self.datasets_list[0].channels_number):
            for j in range(repetitions_arg):
                self.N[j, i] = None
                self.cpm[j, i] = None

        for i in range(self.datasets_list[0].channels_number):
            for j in range(repetitions_arg):
                self.peaks[j, i] = None




        
    
        
#---------------------------------------------------  
        
#---------------------------------------------------

def Fill_datasets_fcs( list_file):

    #print ("Begin")

    current_repetition = 0

    i=0

    channels_fluct_list = []
    channels_cross_list = []
    dataset_list=[]
    full_dataset_list=[]

    array_size_min = -1

    while i < len(list_file):



        if list_file[i].__contains__("CarrierRows"):

            str1 , str2 = list_file[i].split(' = ')
            CarrierRows = int(str2)

            str1 , str2 = list_file[i+1].split(' = ')
            CarrierColumns = int(str2)

            positions = CarrierRows*CarrierColumns


            break

        i +=1



    position = 0
    i = 0

    metadata_filled = False

    while i < len(list_file):

        if list_file[i].__contains__("DetectorWavelengthRangeStart1") and metadata_filled == False:

            list_channel_names = []
            list_starts = []
            list_ends = []

            metadata_filled = True

            channel_index_temp = 1

            while True:

                if list_file[i].__contains__("DetectorWavelengthRangeStart"):

                    list_channel_names.append("channel " + str(channel_index_temp))

                    str1 , str2 = list_file[i].split(' = ')
                    str3 , str4 = str2.split(' ')

                    list_starts.append(float(str3))

                    str1 , str2 = list_file[i+1].split(' = ')
                    str3 , str4 = str2.split(' ')

                    list_ends.append(float(str3))

                    channel_index_temp += 1
                    i += 2

                else:
                    break


            metadata_dict = { "Channel": list_channel_names, "Start": list_starts, "End": list_ends}

            metadata = pd.DataFrame(metadata_dict)






        
        if list_file[i].__contains__("Repetition"):

            

            str1 , str2 = list_file[i].split(' = ')
            repetition = int(str2)

            #print ("Repetition ", repetition)

            if repetition > current_repetition and repetition != -1:

                flag = 0


                dataset_list.append(Dataset_fcs(len(channels_fluct_list), len(channels_cross_list), channels_fluct_list, channels_cross_list))
                current_repetition = repetition
                channels_fluct_list = []
                channels_cross_list = []

            if repetition == current_repetition and repetition != -1:

                str1 , position = list_file[i-2].split(' = ')
                position = int(position)

                str1 , long_name = list_file[i+1].split(' = ')


                

                
                


                if list_file[i+1].__contains__("versus"):

                    str1, str2 = long_name.split(" versus ")
                    str3, str4 = str1.split("Meta")
                    str5, str6 = str2.split("Meta")
                    
                    short_name = "channel " + str4 + " vs " + str(int(str6))

                    str1 , str2 = list_file[i+7].split(' = ')
                    corr_array_size = int(str2)
                    array_corr = list_file[i+9:i+9+corr_array_size]

                    x = []
                    y = []

                    for j in range(len(array_corr)):
                        str1, str2 = array_corr[j].split()
                        x.append(float(str1))
                        y.append(float(str2))

                    array_corr = XY_plot(x,y)

                    channel = fcs_cross(long_name,  array_corr, short_name)

                    channels_cross_list.append(channel)

                    i = i+9+corr_array_size

                    #print (long_name, corr_array_size)



                else:

                    str1, str2 = long_name.split("Meta")
                
                    short_name = "channel " + str(int(str2))
                    

                    str1 , str2 = list_file[i+5].split(' = ')
                    array_size = int(str2)

                    if array_size < array_size_min or array_size_min == -1:
                        array_size_min = array_size

                    array_fluct =list_file[i+7:i+7+array_size]

                    
                   

                    str1 , str2 = list_file[i+7+array_size].split(' = ')
                    corr_array_size = int(str2)
                    

                    #print (long_name, array_size, corr_array_size)

                    array_corr = list_file[i+7+array_size+2:i+7+array_size+2+corr_array_size]

                    x = []
                    y = []

                    for j in range(len(array_fluct)):
                        str1, str2 = array_fluct[j].split()
                        x.append(float(str1))
                        y.append(float(str2))

                    array_fluct = XY_plot(x,y)

                    x = []
                    y = []

                    for j in range(len(array_corr)):
                        str1, str2 = array_corr[j].split()
                        x.append(float(str1))
                        y.append(float(str2))

                    array_corr = XY_plot(x,y)

                    channel = fcs_channel(long_name, array_fluct, array_corr, short_name)

                    channels_fluct_list.append(channel)

                    i = i+7+array_size+2+corr_array_size

                



                i+=1

            if repetition == -1 and flag != 1:

                flag = 1
                dataset_list.append(Dataset_fcs(len(channels_fluct_list), len(channels_cross_list), channels_fluct_list, channels_cross_list))

                repetitions = current_repetition+1

                current_repetition = 0




                for item1 in dataset_list:
                    for item2 in item1.channels_list:
                        del item2.fluct_arr.x[array_size_min-1 : -1]
                        del item2.fluct_arr.y[array_size_min-1 : -1]

                full_dataset = Full_dataset_fcs(repetitions, dataset_list)

                full_dataset.metadata = metadata


                full_dataset.position = str(chr((position)//6 + 65)) + "_" + str((position)%6 + 1)

                full_dataset_list.append(full_dataset)

                #print("position imported: ", full_dataset.position)



                channels_fluct_list = []
                channels_cross_list = []

                dataset_list = []
                

                if position == positions -1: 

                    

                    if positions == 1:
                        full_dataset.position = None

                    


                    
                    
                    break


                i+=1

                continue

        i+=1





    return full_dataset_list

#---------------------------------------------------  
        
#---------------------------------------------------

def Fill_datasets_csv( df, dir_output, filename):


    print("-----------------------------------------")
    print("Fill_datasets_csv")
    print("-----------------------------------------")

    channels_fluct_list = []
    channels_cross_list = []
    dataset_list=[]
    full_dataset_list=[]

    column_t = df.columns[0]

    x = df[column_t]



    data_array = df[df.columns[1:]]

    
    #print(df)
    

    #del data_array[-1]

    data_array = data_array.to_numpy()

    

    if "spp_corrected" not in filename:

        chunk_length = int( 1/(x[1] - x[0]))

        x = list(x)
        del x[-1]

        x_last = x[-1]

        print(x_last)

        timestep = (x[1] - x[0])*chunk_length
        timestep_raw = (x[1] - x[0])

        timepoint = 0

        data_array_new = []
        x_new = []

        print("-----------------------------------------")
        print("Before stitching loop")
        print("timestep: ", chunk_length, len(x), data_array.shape)
        print("-----------------------------------------")

       

        while timepoint + chunk_length < len(x):



            i = int(timepoint)
            j = int(timepoint + chunk_length)


            print(i, j, " of ", len(x))

            #print(arr2)



            sum1 = np.mean(data_array[i:j], axis = 0)

            x_new.append(x[i]/1000)
            data_array_new.append(sum1)

            timepoint = timepoint + chunk_length

            

        arr1 = np.array(data_array_new)

        print("-----------------------------------------")
        print("After stitching loop")
        print("-----------------------------------------")

        #print(arr1)
        print(arr1.shape)

        print("x_new")
        print(len(x_new))

        column_labels = []
        for i in range(arr1.shape[1]):
            column_labels.append("ChS" + str(i+1))

        print(column_labels)

        df_output = pd.DataFrame(arr1, columns=column_labels)

        x_new_arr = np.array(x_new)*1000

        df_output.insert(0, "Time [ms]", x_new)

        print(df_output)

        print(filename)

        excel_file_path = filename + "_spp_corrected.csv"
        df_output.to_csv(excel_file_path, index=False)
    else:


        arr1 = data_array
        x_new = x

    for i in range (len(df.columns) - 1):

        column = df.columns[i+1]

        str1 , str2 = column.split('S')

        long_name = "Auto-correlation detector Meta" + str2
        short_name = "channel " + str(int(str2))

        x = df[column_t]
        y = df[column]

        timestep = (x[1] - x[0])/1000

        x1, y1 = corr_py.correlate_full (timestep, y, y)

        array_corr = XY_plot(x1,y1)

        y_new = arr1[:, i]

        array_fluct = XY_plot(x_new, y_new)


        channel = fcs_channel(long_name, array_fluct, array_corr, short_name)

        channels_fluct_list.append(channel)

    dataset_list.append(Dataset_fcs(len(channels_fluct_list), len(channels_cross_list), channels_fluct_list, channels_cross_list))

    repetitions = 1

    full_dataset = Full_dataset_fcs(repetitions, dataset_list)

    full_dataset_list.append(full_dataset)

    channels_fluct_list = []
    channels_cross_list = []

    dataset_list = []

    full_dataset.position = ""

    return full_dataset_list
    

    #stop_doing_stuff()


    

    #print(len(x), len(y), len(x_new), len(y_new), x_new[1]-x_new[0],  x_new[2]-x_new[1])

        

    """counter = 1
                                i = 0
                                j = 0
                                sum1 = 0 
                        
                                while i < len(x):
                                    sum1 += y[i]
                                    if x[i] > counter:
                                        x_new.append(x[j]/1000)
                                        y_new.append(sum1)
                                        sum1=0
                                        counter += 1
                                        j = i+1
                                    i+=1"""
            
            

        

    """for column1 in df.columns[1:]:
                    for column2 in df.columns[1:]:
                        if column1 != column2:
            
                            str1 , str2 = column1.split('S')
                            str3 , str4 = column2.split('S')
            
                            
                            long_name = "Cross-correlation detector Meta" + str2 + " versus detector Meta" + str4
            
                            str1, str2 = long_name.split(" versus ")
                            str3, str4 = str1.split("Meta")
                            str5, str6 = str2.split("Meta")
                            
                            short_name = "channel " + str4 + " vs " + str(int(str6))
            
                            x = df[column_t]
                            y1 = df[column1]
                            y2 = df[column2]
            
                            timestep = (x[1] - x[0])/1000
            
                            x3, y3 = corr_py.correlate_full (timestep, y1, y2)
            
                            array_corr = XY_plot(x3,y3)
            
                            channel = fcs_cross(long_name,  array_corr, short_name)
            
                            channels_cross_list.append(channel)"""

    
