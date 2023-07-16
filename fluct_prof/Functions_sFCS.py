import tifffile
import matplotlib.pyplot as plt
import numpy as np
import Correlation as corr_py
import lmfit
import pandas as pd
from datetime import datetime
from fluct_prof import Analyse_sFCS_data_MLE_BIC as data_an


class File_sFCS:
    def __init__(self,lsm_file_name):
        self.lsm_file_name = lsm_file_name
        self.array =  tifffile.imread(self.lsm_file_name, key = 0)
        
    def isolate_channel(self,channel_no):
        if len(self.array.shape) == 2:
            return self.array
        else:
            return self.array[channel_no-1]
    
    def spatial_bin(self,channel_no,bin_size):#resulting array has intensities by rows
        channel = self.isolate_channel(channel_no)
        i = 0
        binned_array = np.zeros(len(channel[:,0]))
        while i < len(channel[0])-bin_size+1:
            j = 0
            addition_array = np.zeros(len(channel[:,0]))
            while j < bin_size:
                addition_array += channel[:,i]
                j += 1
                i += 1
            binned_array = np.row_stack((binned_array,addition_array))
        binned_array = np.delete(binned_array,(0),axis=0)
        return binned_array
    
    def slice_in_time(self, channel_no, bin_size, n_slices):
        binned_array = self.spatial_bin(channel_no,bin_size)
        sliced_array = np.zeros(int(len(binned_array[0])/n_slices))
        for i in range(len(binned_array[:,0])):
            arr = np.array_split(binned_array[i],n_slices)
            arr = np.vstack(arr)
            sliced_array = np.row_stack((sliced_array,arr))
        sliced_array = np.delete(sliced_array,(0),axis=0)
        return sliced_array
    
    def intensity_carpet_plot(self,channel_no, bin_size = 1, n_slices = 1):
        binned_data = self.slice_in_time(channel_no, bin_size, n_slices)
        return binned_data
        
    
    def plot_signal(self, channel_no, pixel, bin_size = 1, n_slices = 1): #for binned data, but works for unbinned too!
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        ##Could plot time (partly from from Falk's data):
        scanning_frequency = 2000 #2069 #Hz(global parameter) #from Falk
        line_dur = ((1.0/scanning_frequency)*1000.0) #from Falk
        x = [*range(0,len(binned_array[0]))]
        x = np.array(x)
        time_ms = x * line_dur #gives time in ms
        y = binned_array[pixel]
        plt.figure(figsize=(15,4))
        #plt.plot(x,y)
        plt.plot(time_ms,y)
        plt.ylabel ('Intensity')
        plt.xlabel ('Time (ms)')
        plt.title ('Intensity trace for row no. {}'.format(str(pixel)))
        plt.tight_layout()
        plt.show()    
        return time_ms,y 

    def single_trace(self, channel_no, timestep, row, bin_size = 1, n_slices=1):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        y = binned_array[row]
        x = np.linspace(0, (len(y)-1)*timestep, len(y))
        plt.plot (x, y)
        plt.xlabel('Time (ms)')
        plt.ylabel('Intensity (kHz)')
        plt.tight_layout()
        plt.show()
        return x,y

    
    def single_autoc_plot(self, channel_no, timestep, row, bin_size = 1, n_slices=1):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        y = binned_array[row]
        time, scorr = corr_py.correlate_full (timestep, y, y)
        plt.xscale("log")
        plt.plot (time, scorr)
        plt.xlabel('Delay Time (ms)')
        plt.ylabel('G (tau)')
        plt.tight_layout()
        plt.show()
        return time,scorr

    def autoc_carpet_plot(self, channel_no, timestep, bin_size = 1, n_slices = 1,plot_title =''):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        autocorrelation_by_rows = []
        for i in range(len(binned_array)):
            y = binned_array[i]
            time, scorr = corr_py.correlate_full (timestep, y, y)
            autocorrelation_by_rows.append(scorr)
        fig, ax = plt.subplots(figsize=(100,10))
        im = ax.imshow(autocorrelation_by_rows,origin="lower",cmap='bwr')
        #cbar = ax.figure.colorbar(im, ax=ax,shrink=0.5,location='right', pad =0.003)
        ax.set_title(plot_title)
        plt.show()
        #return autocorrelation_by_rows
    
    def get_fitting_params(self, channel_no, timestep, bin_size, n_slices, 
                           input_params, method='least_squares', export=False):
        binned_array = self.slice_in_time(channel_no, bin_size, n_slices)
        self.params_per_row =[]
        list_of_keys = ["Row_no"]
        for i in input_params.keys():
            list_of_keys.append(i)
        list_of_keys.append('Chi_Sqr')
        self.params_per_row.append(list_of_keys)
        for row in range(len(binned_array)):
            y = binned_array[row]
            time, scorr = corr_py.correlate_full(timestep,y,y)
            o1 = lmfit.minimize(resid,input_params,args=(time,scorr),method=method)
            params_in_row = []
            params_in_row.append(row+1)
            for param in input_params.keys():
                if param == 'txy':
                    params_in_row.append(np.float64(o1.params[param].value)*1000)
                else:
                    params_in_row.append(np.float64(o1.params[param].value))
            params_in_row.append(np.float64(o1.chisqr))
            self.params_per_row.append(params_in_row)
        if export == True:
            export_file_name = "{date}_Ch{channel_no}_{bin_size}bins_{n_slices}slices.csv".format(
                channel_no=channel_no, bin_size=bin_size, n_slices=n_slices, date=datetime.date(datetime.now()))
            self.params_to_csv(export_file_name)
        list_of_keys = self.params_per_row[0]
        param_dict = {}
        for i in list_of_keys:
            param_dict[i]=[]
        for row in range(1,len(self.params_per_row)):
            counter=0
            for i in list_of_keys:
                param_dict[i].append(self.params_per_row[row][counter])
                counter+=1
        self.params_df = pd.DataFrame(param_dict,index=param_dict['Row_no'])
        return self.params_df

    def params_to_csv(self, export_file_name):
        with open (export_file_name,"w") as f:
            for i in self.params_per_row:    
                f.write(','.join([str(k) for k in i]))
                f.write('\n')

    def fitting_figure(self, cutoff_start = 0, cutoff_end=500):
        try:
            cutoff = np.array ([cutoff_start, cutoff_end]) # Upper and lower bounds for transit times. Do not consider too small or too big transit times
            data = self.params_df[self.params_df['txy']< cutoff[1]]
            data = self.params_df[self.params_df['txy']> cutoff[0]]
            results, fig = data_an.model_selection_RL (data['txy'], initial_guess = [4,1], plot = 'on')
            return results
        except:
            print("Error, call File_sFCS.get_fitting_params(....) method before then try again")


def Corr_curve_2d(tc, offset, GN0, A1, txy1, alpha1, B1, tauT1):
    txy1 = txy1
    tauT1 = tauT1
    G_Diff =  A1*(((1+((tc/txy1)**alpha1))**-1))
    G_T = 1 + (B1*np.exp(tc/(-tauT1)))
    return offset + GN0 * G_Diff * G_T

def resid (params, x, ydata ):
    param_list = []    
    for param in params.keys():
        param_list.append( np.float64(params[param].value))
        
    y_model = Corr_curve_2d(x, *param_list)
    return y_model - ydata

def params_lists_to_object(list_of_params, list_of_inits, list_of_vary, list_of_min, list_of_max):
    params = lmfit.Parameters()
    for i in range(0, len(list_of_params)):
        params.add(list_of_params[i], float(list_of_inits[i]), vary = int(list_of_vary[i]), 
                   min = float(list_of_min[i]), max = float(list_of_max[i]))
    return params