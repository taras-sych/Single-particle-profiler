import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import matplotlib.pyplot as plt

import csv

import lmfit

import time


from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib import cm as mplcm

from ttkwidgets import CheckboxTreeview

from fluct_prof import fcs_importer

from fluct_prof import Correlation as corr_py

import codecs

import os

from datetime import datetime

from scipy import stats

import copy

import numpy as np

from scipy.signal import find_peaks

from scipy.optimize import curve_fit
import random

import seaborn as sns

sns.set(context='notebook', style='whitegrid')

global binning_list
binning_list = []

global file_index
global rep_index
global tree_list
tree_list = []

global tree_list_name
tree_list_name = []

global output_file_name

global fit_list_x
global fit_list_y

global Fit_params

global initialdirectory
initialdirectory = ''

global change_normal
change_normal = False


list_of_channel_pairs = []

def Message_generator():
	messages = [
    'You shall not pass!',  
    'Danger!',
    'She is dead, Jim!',
    'My life for the horde!' 
	] 
	index = random.randint(0, len(messages)-1)
	return messages[index]

def Corr_curve_3d(tc, offset, GN0, A1, txy1, alpha1, AR1, B1, tauT1 ):

	txy1 = txy1 / 1000

	tauT1 = tauT1 / 1000

	G_Diff =  (A1*(((1+((tc/txy1)**alpha1))**-1)*(((1+(tc/((AR1**2)*txy1)))**-0.5))))

	G_T = 1 + (B1*np.exp(tc/(-tauT1)))

	return offset + GN0 * G_Diff * G_T

def Corr_curve_2d(tc, offset, GN0, A1, txy1, alpha1, B1, tauT1):

	txy1 = txy1 / 1000

	tauT1 = tauT1 / 1000

	G_Diff =  A1*(((1+((tc/txy1)**alpha1))**-1))

	G_T = 1 + (B1*np.exp(tc/(-tauT1)))

	return offset + GN0 * G_Diff * G_T


def Gauss(x, a, x0, sigma):


	return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def Gauss2(x, a1, x01, sigma1, a2, x02, sigma2):

    return Gauss(x, a1, x01, sigma1) + Gauss(x, a2, x02, sigma2)

def Gauss3(x, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3 ):

    return Gauss(x, a1, x01, sigma1) + Gauss(x, a2, x02, sigma2) + Gauss(x, a3, x03, sigma3)
	
	

def Norm():

	print (1)

def Plot_main():
	global main_xlim
	global main_ylim
	ffp.main.cla()
	#ffp.figure3.clf()

	

	"""print ("Axes before: ", allaxes)
			
				print(len(allaxes))
			
				if len(allaxes) == 2:
					ffp.figure3.delaxes(allaxes[1])
					
			
					#for ax1 in allaxes:
						#ax1.set_aspect('auto')
			
					ffp.figure3.tight_layout()
			
				print ("Axes after: ", allaxes)"""



	list1 = data_frame.tree.get_checked()

	flag = False

	x1 = []
	y1 = []

	for index in list1:
		num1, num = index.split('I')
		

		num = int(num, 16)

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1




		

		file1 = file1-1
		rep1 = rep1-1


		
		
			
		x11 = copy.deepcopy(peaks_list[file1][rep1].x)
		y11 = copy.deepcopy(peaks_list[file1][rep1].y)	

		x1 += x11
		y1 += y11

	
	if ffp.plot_type == "dot plot":
		ffp.main.scatter(x1, y1, color = 'black', s = 1)



	if ffp.plot_type == "density plot":
		my_cmap = mplcm.get_cmap('rainbow')
		my_cmap.set_under('w')
		bins_number = int(np.sqrt(len(x1))/3) 

		min_x1 = min(iiii for iiii in x1 if iiii > 0)

		#print (np.min(x1))

		#print (np.max(x1))

		all_bins_x = np.logspace(np.log10(min_x1),np.log10(np.max(x1)), num=bins_number)

		#all_bins_x = np.logspace(np.log10(min(x1)), np.log10(max(x1)), num=int(np.sqrt(bins_number)),  base=10.0)
		#all_bins_y = np.logspace(np.log10(min(y1)), np.log10(max(y1)), num=int(np.sqrt(bins_number)),  base=10.0)

		#print (all_bins_x)
		#print (all_bins_y)
	

		histogram_tuple = ffp.main.hist2d (x1, y1, bins = [all_bins_x, all_bins_x], cmap = my_cmap, vmin = 5)

		#histogram_tuple = ffp.main.hist2d (x1, y1, bins = 10*bins_number, cmap = my_cmap, vmin = 5)

		

		ffp.main.grid(True)

		

		
		if len(ffp.figure3.get_axes()) == 1:
			ffp.figure3.colorbar(histogram_tuple[3], ax = ffp.main)

		else:
			ffp.figure3.get_axes()[1].cla()
			ffp.figure3.colorbar(histogram_tuple[3], ax = ffp.figure3.get_axes()[1])
		#ffp.figure3.colorbar(pcm, label='Counts')

		

	
	ffp.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
	ffp.main.set_ylabel('Intensity (a.u.)')
	ffp.main.set_xlabel('Intensity (a.u.)')
	ffp.main.set_xscale ('symlog')
	ffp.main.set_yscale ('symlog')
	ffp.main.autoscale(enable=True, axis='both', tight=None)
	ffp.figure3.tight_layout()

	

	

	

def Plot_gp():
	gp_list = []
	data_frame.gp_plot.cla()

	global tree_list_name
	global output_file_name

	list1 = data_frame.tree.get_checked()

	print(list1)

	#print (data_frame.tree.selection())

	thisdict = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		
		print(rep1)

		output_file_name = tree_list_name[file1-1][:-4]
		#print(output_file_name)



		file1 = file1-1
		rep1 = rep1-1




		if data_list_raw[file1].gp_fitting[rep1] != None:

			if len(data_list_raw[file1].gp_fitting[rep1].keys()) == 3:


				if output_file_name in thisdict.keys():

					thisdict[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["Mean"])
				else:
					thisdict[output_file_name] = []
					thisdict[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["Mean"])

			if len(data_list_raw[file1].gp_fitting[rep1].keys()) == 6:

				key = output_file_name + " peak 1"


				if key in thisdict.keys():

					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean1"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean1"])

				key = output_file_name + " peak 2"


				if key in thisdict.keys():

					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean2"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean2"])


			if len(data_list_raw[file1].gp_fitting[rep1].keys()) == 9:

				key = output_file_name + " peak 1"


				if key in thisdict.keys():

					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean1"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean1"])

				key = output_file_name + " peak 2"


				if key in thisdict.keys():

					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean2"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean2"])

				key = output_file_name + " peak 3"


				if key in thisdict.keys():

					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean3"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_list_raw[file1].gp_fitting[rep1]["Mean3"])

		
		
		


	keys = []
	vals = []
	for key in thisdict.keys():
		keys.append(key)
		vals.append(thisdict[key])


	if vals:
		#sns.axlabel( ylabel="GP", fontsize=16)
		sns.boxplot(data=vals, width=.18, ax = data_frame.gp_plot)
		sns.swarmplot(data=vals, size=6, edgecolor="black", linewidth=.9, ax = data_frame.gp_plot)

		# category labels
		data_frame.gp_plot.set_xticklabels(keys)
		#diff.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		data_frame.gp_plot.set_ylabel('GP')


	
	
def Plot_diff():
	diff_list = []
	data_frame.diff_plot.cla()

	global tree_list_name
	global output_file_name

	list1 = data_frame.tree.get_checked()



	thisdict = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		


		output_file_name = tree_list_name[file1-1][:-4]





		file1 = file1-1
		rep1 = rep1-1


		"""if output_file_name in thisdict.keys():
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])
								else:
									thisdict[output_file_name] = []
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])"""

		for item in range(len(data_list_raw[file1].datasets_list[rep1].channels_list)):

			if data_frame.channels_flags[data_list_raw[file1].datasets_list[rep1].channels_list[item].short_name].get() == 1 and data_list_raw[file1].diff_fitting[rep1, item]!= None:

				key = output_file_name + " " + data_list_raw[file1].datasets_list[rep1].channels_list[item].short_name 

				if key in thisdict.keys():
					thisdict[key].append(data_list_raw[file1].diff_fitting[rep1, item]["txy"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_list_raw[file1].diff_fitting[rep1, item]["txy"])

		for i in range(len(data_list_raw[file1].datasets_list[rep1].cross_list)):

			item = i + len(data_list_raw[file1].datasets_list[rep1].channels_list)



			if data_frame.channels_flags[data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name].get() == 1 and data_list_raw[file1].diff_fitting[rep1, item] != None:

				key = output_file_name + " " + data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name 

				if key in thisdict.keys():
					thisdict[key].append(data_list_raw[file1].diff_fitting[rep1, item]["txy"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_list_raw[file1].diff_fitting[rep1, item]["txy"])

		



	keys = []
	vals = []
	for key in thisdict.keys():
		keys.append(key)
		vals.append(thisdict[key])


	if vals:
		sns.set(context='notebook', style='whitegrid')
		#sns.axlabel( ylabel="Diffusion time", fontsize=16)
		sns.boxplot(data=vals, width=.18, ax = data_frame.diff_plot)
		sns.swarmplot(data=vals, size=6, edgecolor="black", linewidth=.9, ax = data_frame.diff_plot)

		# category labels
		data_frame.diff_plot.set_xticklabels(keys)
		#diff.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		data_frame.diff_plot.set_ylabel('Diffusion time')
		

def Plot_gp_diff():

	gp_diff.main.cla()

	global tree_list_name
	global output_file_name

	list1 = data_frame.tree.get_checked()



	thisdict_gp = {}
	thisdict_diff = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		


		output_file_name = tree_list_name[file1-1][:-4]




		file1 = file1-1
		rep1 = rep1-1


		"""if output_file_name in thisdict.keys():
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])
								else:
									thisdict[output_file_name] = []
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])"""

		if output_file_name in thisdict_diff.keys():
			thisdict_diff[output_file_name].append(data_list_raw[file1].diff_fitting[rep1, 0]["txy"])
		else:
			thisdict_diff[output_file_name] = []
			thisdict_diff[output_file_name].append(data_list_raw[file1].diff_fitting[rep1, 0]["txy"])


		if output_file_name in thisdict_gp.keys():
			thisdict_gp[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["Mean"])
		else:
			thisdict_gp[output_file_name] = []
			thisdict_gp[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["Mean"])

		
	for key in thisdict_diff.keys():
		gp_diff.main.scatter(thisdict_gp[key], thisdict_diff[key], label = key )
		gp_diff.main.legend(loc='upper right')

	gp_diff.main.set_ylabel('Diffusion time (ms)')
	gp_diff.main.set_xlabel('GP')






def Which_tab():



	Plot_diff()


	Plot_gp()

	data_frame.canvas1.draw()
	data_frame.figure1.tight_layout()



	#if tabs.index(tabs.select()) == 2:
		#Plot_gp_diff()

		#gp_diff.canvas3.draw()

		#gp_diff.figure3.tight_layout()

	#except:
		#tk.messagebox.showerror(title='Error', message=Message_generator())



def Threshold_fun():

	if len(tree_list_name) > 0:

		th_win = Threshold_window(win_width, win_height, dpi_all)

	if len(tree_list_name) == 0:

		tk.messagebox.showerror(title='Error', message=Message_generator())

def Diffusion_fun():
	if len(tree_list_name) > 0:

		th_win = Diffusion_window(win_width, win_height, dpi_all)

	if len(tree_list_name) == 0:

		tk.messagebox.showerror(title='Error', message=Message_generator())

def Dot_Plot_fun():

	if len(tree_list_name) > 0:

		dot_plot_win = Dot_Plot_Window(win_width, win_height, dpi_all)

	if len(tree_list_name) == 0:

		tk.messagebox.showerror(title='Error', message=Message_generator())


def Restruct_fun():


	th_win = Restruct_window(win_width, win_height, dpi_all)


def Export_function():


	global tree_list_name
	global output_file_name
	global initialdirectory

	now = datetime.now()
	str1, str2 = str(now).split(".")
	name_dir = str1 + " Analysis"

	name_dir = name_dir.replace(":", "_")

	initialdirectory

	directory = os.path.join(initialdirectory, name_dir) 
    

	os.mkdir(directory) 




	output_numbers_dict = {}





	list1 = data_frame.tree.get_checked()





	thisdict_gp = {}
	thisdict_diff = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		file1 = file1-1
		rep1 = rep1-1

		if file1 in output_numbers_dict.keys():
			output_numbers_dict[file1].append(rep1)
		else:
			output_numbers_dict[file1] = []
			output_numbers_dict[file1].append(rep1)

	summary_diffusion_dict = {}
	summary_gp_dict = {}


	#filename = directory + os.path.sep + "Summary_Diffusion.csv"
	#summary_diffusion_file = open(filename, 'w')

	#filename = directory + os.path.sep + "Summary_Diffusion_Time.csv"
	#summary_time_file = open(filename, 'w')

	#filename = directory + os.path.sep + "Summary_GP.csv"
	#summary_GP_file = open(filename, 'w')

	#filename = directory + os.path.sep + "Clustering.csv"
	#clustering_file = open(filename, 'w')

	heading = ""
	heading_line_1 = ""
	heading_line_2 = ""

	for file1 in output_numbers_dict.keys():

		heading += str(file1)

		heading_line_1 += str(file1) + "\t\t\t"

		heading_line_2 += "GP" + "\t" + "Diffusion_coef" + "\t" + "Diffusion_time" + "\t"

		output_file_name, str2 = tree_list_name[file1].split(".")



		filename = directory + os.path.sep + output_file_name + ".txt"

		open_file = open (filename, "w")

		open_file.write(output_file_name + "\n")

		open_file.write("Diffusion data: \n")

		chan = {}


		for channel in range(len(data_list_raw[file1].datasets_list[rep1].channels_list)):

			if data_frame.channels_flags[data_list_raw[file1].datasets_list[rep1].channels_list[channel].short_name].get() == 1:

				chan[data_list_raw[file1].datasets_list[rep1].channels_list[channel].short_name] = channel


		for i in range (len(data_list_raw[file1].datasets_list[rep1].cross_list)):

			channel = i + len(data_list_raw[file1].datasets_list[rep1].channels_list)

			if data_frame.channels_flags[data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name].get() == 1:

				chan[data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name] = channel

		
		for name in chan.keys():

			channel = chan[name]
			open_file.write(name + "\n")

			line = "name\t"

			rep0 = output_numbers_dict[file1][0]

			if data_list_raw[file1].diff_fitting[rep0, channel] != None:

				for key in data_list_raw[file1].diff_fitting[rep0, channel].keys():

					line += key + "\t"

				line += "N" + "\t"
				line += "cpm" + "\t"
				line += "D" + "\t"


			open_file.write(line + "\n")



			for rep1 in output_numbers_dict[file1]:

				line = "Repetition " + str(rep1 + 1) + "\t"

				try:

					for key in data_list_raw[file1].diff_fitting[rep1, channel].keys():

						line += str(data_list_raw[file1].diff_fitting[rep1, channel][key]) + "\t"
				except: 
					pass

				
				try:
					line += str(data_list_raw[file1].N[rep1, channel]) + "\t"
					line += str(data_list_raw[file1].cpm[rep1, channel]) + "\t"
				except:
					pass

				
				try:
					line += str(data_list_raw[file1].diff_coeffs[rep1, channel]) + "\t"
				except:
					pass


				open_file.write(line + "\n")


		open_file.write("\n")
		open_file.write("GP data: \n")

		rep0 = output_numbers_dict[file1][0]

		line = "name\t"

		try:

			for key in data_list_raw[file1].gp_fitting[rep0].keys():
				line += key + "\t"
		except:
			pass

		open_file.write(line + "\n")

		for rep1 in output_numbers_dict[file1]:

			line = "Repetition " + str(rep1 + 1) + "\t"

			try:

				for key in data_list_raw[file1].gp_fitting[rep1].keys():

					line += str(data_list_raw[file1].gp_fitting[rep1][key]) + "\t"

			except:
				pass


			open_file.write(line + "\n")





		open_file.close()




class Restruct_window:

	def Temp(self):
		print ("It is temp")

	def Restructure_dataset(self):
		global file_index
		global rep_index

		temp_dict = {}

		for channel in range (data_list_raw[file_index].datasets_list[rep_index].channels_number):

			x = []
			y = []

			temp_dict[channel] = fcs_importer.XY_plot(x,y)


		

			for rep_index_i in range (data_list_raw[file_index].repetitions):
								
					

				#print ("adding repetition ", rep_index_i)


				if len(temp_dict[channel].x) == 0:
					x_min = 0
				else:
					x_min = max(temp_dict[channel].x) + temp_dict[channel].x[1] - temp_dict[channel].x[0]

				x_temp_1 = [elem + x_min for elem in data_list_raw[file_index].datasets_list[rep_index_i].channels_list[channel].fluct_arr.x]

				temp_dict[channel].x.extend(x_temp_1)

				temp_dict[channel].y.extend(data_list_raw[file_index].datasets_list[rep_index_i].channels_list[channel].fluct_arr.y)



		repetitions_new = int(self.num_rep.get())


		length_rep = int (len (temp_dict[0].x)/repetitions_new)
		

		
		dataset_list_arg = []
		for rep_index_i in range (repetitions_new):


			channels_list_arg = []

			for channel in range (data_list_raw[file_index].datasets_list[rep_index].channels_number):

				end = length_rep*(rep_index_i + 1)
				start = end - length_rep

				if rep_index_i == repetitions_new-1:

					if end != len (temp_dict[0].x) - 1:

						end = len (temp_dict[0].x) - 1

				

				x = temp_dict[channel].x[start : end]
				y = temp_dict[channel].y[start : end]

				min1 = min(x)

				x1 = [a - min1 for a in x]

				x = x1

				

				Tr = fcs_importer.XY_plot(x,y)

				timestep = x[1] - x[0]

				x1, y1 = corr_py.correlate_full (timestep, np.array(Tr.y), np.array(Tr.y))

				AutoCorr = fcs_importer.XY_plot(x1,y1)

				long_name = data_list_raw[file_index].datasets_list[rep_index].channels_list[channel].name

				short_name = data_list_raw[file_index].datasets_list[rep_index].channels_list[channel].short_name

				Ch_dataset = fcs_importer.fcs_channel (long_name, Tr, AutoCorr, short_name)

				channels_list_arg.append(Ch_dataset)

			FCS_Dataset =  fcs_importer.Dataset_fcs(data_list_raw[file_index].datasets_list[rep_index].channels_number, 0, channels_list_arg, [] )

			dataset_list_arg.append(FCS_Dataset)



		
		dataset = 	fcs_importer.Full_dataset_fcs(repetitions_new, dataset_list_arg)

		name = tree_list_name[file_index] + " " + str(repetitions_new)

		treetree = Data_tree (self.tree, name, dataset.repetitions)

		treetree = Data_tree (data_frame.tree, name, dataset.repetitions)

		tree_list.append(treetree)

		tree_list_name.append(name)

		binning_list.append(1)


		data_list_raw.append(dataset)


		#data_list_current.append(dataset1)


		total_channels_list.append(dataset.datasets_list[0].channels_number + dataset.datasets_list[0].cross_number)
		repetitions_list.append(dataset.repetitions)

		peaks_list.append([None] * dataset.repetitions)

		list_of_channel_pairs.append([None])






	def Plot_curve(self):


		global file_index
		global rep_index

		
		self.curves.cla()
		self.traces.cla()



		for item in data_list_raw[file_index].datasets_list[rep_index].channels_list:

			x1 = item.auto_corr_arr.x
			y1 = item.auto_corr_arr.y

		
			self.curves.plot(x1, y1, label = item.short_name)

		if data_list_raw[file_index].datasets_list[rep_index].cross_number > 0:
			for item in data_list_raw[file_index].datasets_list[rep_index].cross_list:

				x1 = item.cross_corr_arr.x
				y1 = item.cross_corr_arr.y

			
				self.curves.plot(x1, y1, label = item.short_name)



		
		
		self.curves.set_title("Correlation curves")
		self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.curves.set_ylabel('G(tau)')
		self.curves.set_xlabel('Delay time')
		self.curves.set_xscale ('log')


		for item in data_list_raw[file_index].datasets_list[rep_index].channels_list:


			x1 = item.fluct_arr.x
			y1 = item.fluct_arr.y

		
			self.traces.plot(x1, y1, label = item.short_name)




		self.traces.set_title("Intensity traces")
		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('Intensity')
		self.traces.set_xlabel('Time (s)')



		self.traces.legend(loc='upper right')

		

		self.curves.legend(loc='upper right')

		self.canvas5.draw()

		self.figure5.tight_layout()



	def Choose_curve(self, event):

		global file_index
		global rep_index

		index = self.tree.selection()
		num1, num = index[0].split('I')


		

		num = int(num, 16)

		sum1 = num 
		file = 0

		rep = 0
		


		for i in range (len(data_list_raw)):
			#print ("I am here")
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1




		

		file_index = file1-1
		rep_index = rep1-1



		self.Plot_curve()




	def __init__(self, win_width, win_height, dpi_all):


		global file_index
		global rep_index

		self.win_diff = tk.Toplevel()

		self.th_width = round(0.7*self.win_diff.winfo_screenwidth())
		self.th_height = round(0.4*self.win_diff.winfo_screenwidth())

		self.line1 = str(self.th_width) + "x" + str(self.th_height)


		self.win_diff.geometry(self.line1)

		self.frame002 = tk.Frame(self.win_diff)
		self.frame002.pack(side = "left", anchor = "nw")


		self.frame003 = tk.Frame(self.frame002)
		self.frame003.pack(side = "top", anchor = "nw")		



		self.scrollbar = tk.Scrollbar(self.frame003)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame003, width = 100, height = 10)
		self.Datalist.pack(side = "top", anchor = "nw")
		
		
		
		self.tree = CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Choose_curve)



		self.Datalist.config(width = 100, height = 10)


		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		Label_1 = tk.Label(self.frame004, text="Repetitions: ")
		Label_1.grid(row = 0, column = 0, sticky = 'w')



		self.num_rep = tk.Entry(self.frame004, width = 9)
		self.num_rep.grid(row = 0, column = 1, sticky='w')

		self.num_rep.delete(0,"end")
		self.num_rep.insert(0,data_list_raw[file_index].repetitions)

		self.Rep_button = tk.Button(self.frame004, text="Apply reps", command=self.Restructure_dataset)
		self.Rep_button.grid(row = 0, column = 2, rowspan = 2, sticky='w')

		Label_2 = tk.Label(self.frame004, text="Each rep: ")
		Label_2.grid(row = 1, column = 0, sticky = 'w')

		text1 = str(round(data_list_raw[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.x[-1],1)) + "sec"

		Label_3 = tk.Label(self.frame004, text=text1)
		Label_3.grid(row = 1, column = 1, sticky = 'w')

		self.Remove_button = tk.Button(self.frame004, text="Remove dataset", command=self.Temp)
		self.Remove_button.grid(row = 2, column = 0, columnspan = 3, sticky = 'ew')



		self.frame000 = tk.Frame(self.win_diff)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi = dpi_all)
						
		gs = self.figure5.add_gridspec(2, 1)


		self.traces = self.figure5.add_subplot(gs[:1, 0])

		self.traces.set_title("Correlation curves")

		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('G(tau)')
		self.traces.set_xlabel('Delay time (s)')

		self.curves = self.figure5.add_subplot(gs[1, 0])

		#self.hist1.set_title("Intensity histogram")

		self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.curves.set_ylabel('Counts')
		self.curves.set_xlabel('Residuals')




		self.canvas5 = FigureCanvasTkAgg(self.figure5, self.frame000)
		self.canvas5.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas5, self.frame000)
		self.toolbar.update()
		self.canvas5.get_tk_widget().pack()

		self.figure5.tight_layout()

		for i in range(0, len(tree_list_name)):
			name = tree_list_name[i]
			treetree = Data_tree (self.tree, name, data_list_raw[i].repetitions)

		self.tree.selection_set(treetree.child_id)
		





class Left_frame :







	def Plot_this_data(self, datasets_pos, rep):

		

		

		self.traces.cla()

		self.corr.cla()



		for i in range (datasets_pos.datasets_list[rep].channels_number): 

			

			if self.channels_flags[datasets_pos.datasets_list[rep].channels_list[i].short_name].get() == 1:

				self.traces.plot(datasets_pos.datasets_list[rep].channels_list[i].fluct_arr.x, datasets_pos.datasets_list[rep].channels_list[i].fluct_arr.y, label = datasets_pos.datasets_list[rep].channels_list[i].short_name)

				self.corr.plot(datasets_pos.datasets_list[rep].channels_list[i].auto_corr_arr.x, datasets_pos.datasets_list[rep].channels_list[i].auto_corr_arr.y, label = datasets_pos.datasets_list[rep].channels_list[i].short_name)

		for i in range (datasets_pos.datasets_list[rep].cross_number):

			if self.channels_flags[datasets_pos.datasets_list[rep].cross_list[i].short_name].get() == 1:

				self.corr.plot(datasets_pos.datasets_list[rep].cross_list[i].cross_corr_arr.x, datasets_pos.datasets_list[rep].cross_list[i].cross_corr_arr.y, label = datasets_pos.datasets_list[rep].cross_list[i].short_name)


		


		
		

		self.traces.set_title("Intensity traces")
		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('Counts (Hz)')
		self.traces.set_xlabel('Time (s)')
		self.traces.legend(loc='upper right')




		self.corr.set_title("Correlation curves")
		self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.corr.set_ylabel('G(tau)')
		self.corr.set_xlabel('Delay time')
		self.corr.set_xscale ('log')
		self.corr.legend(loc='upper right')


		self.canvas1.draw_idle()

		

		self.figure1.tight_layout()

		
		
		




	def Import(self):

		


		

		global tree_list
		global tree_list_name
		global initialdirectory

		if initialdirectory == '':
			initialdirectory = __file__

		ftypes = [('FCS .fcs', '*.fcs'), ('FCS .SIN', '*.SIN'), ('Text files', '*.txt'), ('All files', '*'), ]
		

		filenames =  tk.filedialog.askopenfilenames(initialdir=os.path.dirname(initialdirectory),title = "Select file", filetypes = ftypes)

		
		filename = filenames[0]
		#print (filename)

		line = "file 1 out of " + str(len(filenames))

		self.pb = ttk.Progressbar(self.framepb, orient='horizontal', mode='determinate', length=280)
		self.pb.pack(side = "left", anchor = "nw")
		self.value_label = ttk.Label(self.framepb, text=line)
		self.value_label.pack(side = "left", anchor = "nw")

		for filename_index in range (0, len(filenames)):
			filename = filenames[filename_index]
			if filename != "":

				self.pb['value'] = (filename_index+1)/len(filenames) * 100
				self.value_label['text'] = "file " + str(filename_index + 1) + " out of " + str(len(filenames))

				initialdirectory = os.path.dirname(filename)

				

				#progress_window.grab_set()


				name = os.path.basename(filename)

				file = codecs.open (filename, encoding='latin')

				lines = file.readlines()

				if filename.endswith('.fcs'):
					dataset = fcs_importer.Fill_datasets_fcs(lines)

				if filename.endswith('.SIN'): 
					dataset = fcs_importer.Fill_datasets_sin(lines)

				#dataset1 = copy.deepcopy(dataset)


				treetree = Data_tree (self.tree, name, dataset.repetitions)
				self.tree.selection_set(treetree.child_id)
				tree_list.append(treetree)

				tree_list_name.append(name)

				binning_list.append(1)


				data_list_raw.append(dataset)


				#data_list_current.append(dataset1)


				total_channels_list.append(dataset.datasets_list[0].channels_number + dataset.datasets_list[0].cross_number)
				repetitions_list.append(dataset.repetitions)

				peaks_list.append([None] * dataset.repetitions)

				list_of_channel_pairs.append([None])

				root.update() 



		self.pb.destroy()
		self.value_label.destroy()



	def Select_Unselect(self):

		global file_index
		global rep_index

		self.Plot_this_data(data_list_raw[file_index], rep_index)

		root.update()


	def Plot_data(self, event):

		start = time.time()

		global file_index
		global rep_index

		index = self.tree.selection()
		num1, num = index[0].split('I')


		

		num = int(num, 16)

		sum1 = num 
		file = 0

		rep = 0
		


		for i in range (len(data_list_raw)):
			#print ("I am here")
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1




		

		file_index = file1-1
		rep_index = rep1-1


		

		

		rep = rep1-1


		data_frame.Curve_flags()

		

		self.Plot_this_data(data_list_raw[file_index], rep)

		root.update()

	def Delete_dataset(self):
		global file_index
		index = self.tree.selection()
		for sel in index:
			self.tree.delete(sel)

	def Delete_all_datasets(self):
		global data_list_raw
		global data_list_current
		global tree_list
		global tree_list_name



		for dataset in self.tree.get_children():
			self.tree.delete(dataset)
		self.traces.clear()
		self.corr.clear()
		self.canvas1.draw_idle()
	
		self.figure1.tight_layout()
	

		data_list_raw = []
		data_list_current = []
		tree_list = []
		tree_list_name = []


	def Curve_flags(self):

		self.frame0003.destroy()

		self.frame0003 = tk.Frame(self.frame024)
		self.frame0003.pack(side = "left", anchor = "nw")

		self.flags_dict = {}
		self.channels_flags = {}
		self.cross_flags = []
		column_counter = 0

		channels_to_display = 0

		for i in range (len(data_list_raw)):
			if data_list_raw[i].datasets_list[0].channels_number > channels_to_display:
				channels_to_display = data_list_raw[i].datasets_list[0].channels_number
				file_index_local = i


		for item in data_list_raw[file_index_local].datasets_list[rep_index].channels_list:
			str1, str2 = item.short_name.split(" ")
			very_short_name = "ch0" + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command = self.Select_Unselect)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1

		for item in data_list_raw[file_index_local].datasets_list[rep_index].cross_list:
			str1, str2 = item.short_name.split(" vs ")
			str3, str4 = str1.split(" ")
			very_short_name = "ch" + str4 + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command = self.Select_Unselect)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1


	def __init__ (self, frame0, win_width, win_height, dpi_all):


		

		self.frame01 = tk.Frame(frame0)
		self.frame01.pack(side="top", fill="x")


		self.Import_Button = tk.Button(self.frame01, text="Import", command=self.Import)
		self.Import_Button.pack(side = "left", anchor = "nw")

		self.Clear_Button = tk.Button(self.frame01, text="Delete dataset", command=self.Delete_dataset)
		self.Clear_Button.pack(side = "left", anchor = "nw")

		self.Clear_all_Button = tk.Button(self.frame01, text="Delete all", command=self.Delete_all_datasets)
		self.Clear_all_Button.pack(side = "left", anchor = "nw")


		self.frame02 = tk.Frame(frame0)
		self.frame02.pack(side="left", fill="x", anchor = "nw")

		self.frame04 = tk.Frame(frame0)
		self.frame04.pack(side="left", fill="x", anchor = "nw")


		self.frame03 = tk.Frame(self.frame02)
		self.frame03.pack(side="top", fill="x")



		self.scrollbar = tk.Scrollbar(self.frame03)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame03, width = 150, height = 10)
		self.Datalist.pack(side = "left", anchor = "nw")
		
		
		
		self.tree=CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Plot_data)

		self.Datalist.config(width = 100, height = 10)

		self.frame024 = tk.Frame(self.frame02)
		self.frame024.pack(side = "top", fill = "x", anchor='nw')

		self.frame0003 = tk.Frame(self.frame024)
		self.frame0003.pack(side = "left", fill = "x")


		#self.chkbtn = tk.Checkbutton(self.frame0003, text="ch1", variable=1, command=Norm)
		#self.chkbtn.grid(row = 0, column = 0, sticky='w')

		self.frame023 = tk.Frame(self.frame02)
		self.frame023.pack(side="left", fill="x")


		self.Restruct_button = tk.Button(self.frame023, text="Restructure data", command=Restruct_fun)
		self.Restruct_button.grid(row = 0, column = 0, sticky="EW")

		self.Threshold_button = tk.Button(self.frame023, text="Peak analysis", command=Threshold_fun)
		self.Threshold_button.grid(row = 1, column = 0, sticky="EW")

		self.Diffusion_button = tk.Button(self.frame023, text="Diffusion analysis", command=Diffusion_fun)
		self.Diffusion_button.grid(row = 2, column = 0, sticky="EW")

		self.Add_to_plot_button = tk.Button(self.frame023, text="Plot", command=Which_tab)
		self.Add_to_plot_button.grid(row = 3, column = 0, sticky="EW")

		
		self.Add_to_plot_button = tk.Button(self.frame023, text="Dot Plot", command=Dot_Plot_fun)
		self.Add_to_plot_button.grid(row = 4, column = 0, sticky="EW")

		self.Output_button = tk.Button(self.frame023, text="Output", command=Export_function)
		self.Output_button.grid(row = 5, column = 0, sticky="EW")

		self.figure1 = Figure(figsize=(0.85*win_height/dpi_all,0.85*win_height/dpi_all), dpi = dpi_all)




		gs = self.figure1.add_gridspec(3, 2)


		self.traces = self.figure1.add_subplot(gs[:1, :2])

		self.traces.set_title("Intensity traces")

		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('intensity (a.u.)')
		self.traces.set_xlabel('Time (s)')

		


		self.corr = self.figure1.add_subplot(gs[1, :2])

		self.corr.set_title("Correlation curves")

		self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.corr.set_ylabel('G (tau)')
		self.corr.set_xlabel('Delay time')


		self.diff_plot = self.figure1.add_subplot(gs[2, 0])

		self.diff_plot.set_title("Diffusion")
		self.diff_plot.set_ylabel('Diff. Coeff.')
		



		self.gp_plot = self.figure1.add_subplot(gs[2, 1])

		self.gp_plot.set_title("General Polarization")
		self.gp_plot.set_ylabel('GP')





		self.canvas1 = FigureCanvasTkAgg(self.figure1, self.frame04)
		self.canvas1.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas1, self.frame04)
		self.toolbar.update()
		self.canvas1.get_tk_widget().pack()

		self.figure1.tight_layout()

		self.framepb = tk.Frame(frame0)
		self.framepb.pack(side="top", fill="x")







	
class Diff_frame :


	def Curve_flags(self):

		self.frame0003.destroy()

		self.frame0003 = tk.Frame(self.frame13)
		self.frame0003.pack(side = "top", anchor = "nw")

		self.flags_dict = {}
		self.channels_flags = {}
		self.cross_flags = []
		column_counter = 0

		channels_to_display = 0

		for i in range (len(data_list_raw)):
			if data_list_raw[i].datasets_list[0].channels_number > channels_to_display:
				channels_to_display = data_list_raw[i].datasets_list[0].channels_number
				file_index_local = i




		for item in data_list_raw[file_index_local].datasets_list[rep_index].channels_list:
			str1, str2 = item.short_name.split(" ")
			very_short_name = "ch0" + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command=Norm)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1

		for item in data_list_raw[file_index_local].datasets_list[rep_index].cross_list:
			str1, str2 = item.short_name.split(" vs ")
			str3, str4 = str1.split(" ")
			very_short_name = "ch" + str4 + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command=Norm)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1

	def __init__ (self, frame1, win_width, win_height, dpi_all):

		
		self.frame13 = tk.Frame(frame1)
		self.frame13.pack(side="top", fill="x")

		self.frame0003 = tk.Frame(self.frame13)
		self.frame0003.pack(side="top", fill="x")


		self.frame12 = tk.Frame(frame1)
		self.frame12.pack(side="top", fill="x")


		self.figure3 = Figure(figsize=(win_width/(4*dpi_all),win_width/(4.25*dpi_all)), dpi = dpi_all)
		self.main = self.figure3.add_subplot(1, 1, 1)

		self.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.main.set_ylabel('Counts')
		self.main.set_xlabel('GP')


		self.canvas3 = FigureCanvasTkAgg(self.figure3, self.frame12)
		self.canvas3.get_tk_widget().pack()


		self.toolbar = NavigationToolbar2Tk(self.canvas3, self.frame12)
		self.toolbar.update()
		self.canvas3.get_tk_widget().pack()

		self.figure3.tight_layout()

		self.frame13 = tk.Frame(frame1)
		self.frame13.pack(side="top", fill="x")


		
class GP_frame :



	def __init__(self, frame1, win_width, win_height, dpi_all):

		

		
		



		self.frame12 = tk.Frame(frame1)
		self.frame12.pack(side="top", fill="x")


		self.figure3 = Figure(figsize=(win_width/(2*dpi_all),win_width/(2.25*dpi_all)), dpi = dpi_all)
		self.main = self.figure3.add_subplot(1, 1, 1)

		self.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.main.set_ylabel('Counts')
		self.main.set_xlabel('GP')


		self.canvas3 = FigureCanvasTkAgg(self.figure3, self.frame12)
		self.canvas3.get_tk_widget().pack()


		self.toolbar = NavigationToolbar2Tk(self.canvas3, self.frame12)
		self.toolbar.update()
		self.canvas3.get_tk_widget().pack()

		self.figure3.tight_layout()

		self.frame13 = tk.Frame(frame1)
		self.frame13.pack(side="top", fill="x")



class GP_Diff_frame:


	def __init__ (self, frame1, win_width, win_height, dpi_all):

		self.frame12 = tk.Frame(frame1)
		self.frame12.pack(side="top", fill="x")


		self.figure3 = Figure(figsize=(win_width/(2*dpi_all),win_width/(2.25*dpi_all)), dpi = dpi_all)
		self.main = self.figure3.add_subplot(1, 1, 1)

		self.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.main.set_ylabel('Diffusion time (ms)')
		self.main.set_xlabel('GP')


		self.canvas3 = FigureCanvasTkAgg(self.figure3, self.frame12)
		self.canvas3.get_tk_widget().pack()


		self.toolbar = NavigationToolbar2Tk(self.canvas3, self.frame12)
		self.toolbar.update()
		self.canvas3.get_tk_widget().pack()

		self.figure3.tight_layout()

		self.frame13 = tk.Frame(frame1)
		self.frame13.pack(side="top", fill="x")



class Diffusion_window :

	def Save_plot_data(self):
		filename = initialdirectory + "\\Plots_diffusion.txt"

		open_file = open(filename, 'w')

		for key in self.save_plot_dict.keys():
			open_file.write(str(key) + "\n")

			for i in range(len(self.save_plot_dict[key].x)):
				open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

		open_file.close()

	def Apply_to_all(self):
	
		global rep_index

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()



		for rep_index_i in range (data_list_raw[file_index].repetitions): 
			rep_index = rep_index_i
			for channel_index_i in range(data_list_raw[file_index].datasets_list[rep_index].channels_number + data_list_raw[file_index].datasets_list[rep_index].cross_number):


				if channel_index_i < data_list_raw[file_index].datasets_list[rep_index].channels_number:
					
					if self.channels_flags[channel_index_i].get() == 1:

						self.channel_index = channel_index_i
						for param in self.list_of_params:
							self.full_dict[param]["Init"].delete(0,"end")
							self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))

						self.Fit_corr_curve()


				else:

					if self.cross_flags[channel_index_i - data_list_raw[file_index].datasets_list[rep_index].channels_number].get() == 1:

						self.channel_index = channel_index_i
						for param in self.list_of_params:
							self.full_dict[param]["Init"].delete(0,"end")
							self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))

						self.Fit_corr_curve()


					#self.Plot_curve()
					

		

		self.fit_all_flag = False

		self.Plot_curve()


	def Apply_to_ticked(self):

		#print("Apply to ticked")

		global file_index
		global rep_index
		#self.curve_index = 0

		list1 = self.tree.get_checked()


		for index in list1:

			num1, num = index.split('I')


			

			num = int(num, 16)

			sum1 = num 
			file = 0

			rep = 0

			ch = 0
			
			

			for i in range (len(data_list_raw)):
				#print ("I am here")
				rep = 0
				ch = 0
				sum1-=1
				file+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep
					ch1 = ch
				
				
				for j in range (repetitions_list[i]):
					ch = 0
					sum1-=1
					
					rep+=1
					if sum1 == 0:
						file1 = file
						rep1 = rep
						ch1 = ch

					for k in range (total_channels_list[i]):
						sum1-=1

						ch+=1
						if sum1 == 0:
							file1 = file
							rep1 = rep
							ch1 = ch






			if rep1 == 0:
				rep1+=1

			if ch1 == 0:
				ch1+=1

			file_index = file1-1
			
			rep_index = rep1-1

			self.channel_index = ch1-1
			

			self.fit_all_flag = True

			self.list_of_inits_for_fit_all = {}

			for param in self.list_of_params:
				self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()



			 
			
			channel_index_i = self.channel_index


			if channel_index_i < data_list_raw[file_index].datasets_list[rep_index].channels_number:
				
				if self.channels_flags[channel_index_i].get() == 1:

					self.channel_index = channel_index_i
					for param in self.list_of_params:
						self.full_dict[param]["Init"].delete(0,"end")
						self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))

					self.Fit_corr_curve()


			else:

				if self.cross_flags[channel_index_i - data_list_raw[file_index].datasets_list[rep_index].channels_number].get() == 1:

					self.channel_index = channel_index_i
					for param in self.list_of_params:
						self.full_dict[param]["Init"].delete(0,"end")
						self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))

					self.Fit_corr_curve()


					#self.Plot_curve()
					

		

		self.fit_all_flag = False

		self.Plot_curve()

	def Apply_to_all_all(self):
				

		global rep_index
		global file_index

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()



		for file_index_i in range (len(data_list_raw)):
			file_index = file_index_i
			for rep_index_i in range (data_list_raw[file_index].repetitions): 
				rep_index = rep_index_i
				for channel_index_i in range(data_list_raw[file_index].datasets_list[rep_index].channels_number + data_list_raw[file_index].datasets_list[rep_index].cross_number):


					if channel_index_i < data_list_raw[file_index].datasets_list[rep_index].channels_number:
						
						if self.channels_flags[channel_index_i].get() == 1:

							self.channel_index = channel_index_i
							for param in self.list_of_params:
								self.full_dict[param]["Init"].delete(0,"end")
								self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))

							self.Fit_corr_curve()


					else:

						if self.cross_flags[channel_index_i - data_list_raw[file_index].datasets_list[rep_index].channels_number].get() == 1:

							self.channel_index = channel_index_i
							for param in self.list_of_params:
								self.full_dict[param]["Init"].delete(0,"end")
								self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))

							self.Fit_corr_curve()



					

		

		self.fit_all_flag = False

		self.Plot_curve()


	def Fit_corr_curve(self):



		
		if self.channel_index < data_list_raw[file_index].datasets_list[rep_index].channels_number:
			#print('Channel to process', self.channel_index)
			x = data_list_raw[file_index].datasets_list[rep_index].channels_list[self.channel_index].auto_corr_arr.x
			y = data_list_raw[file_index].datasets_list[rep_index].channels_list[self.channel_index].auto_corr_arr.y

		else:

			num = data_list_raw[file_index].datasets_list[rep_index].channels_number
			#print('Cross to process', num - self.channel_index)
			x = data_list_raw[file_index].datasets_list[rep_index].cross_list[self.channel_index - num].cross_corr_arr.x
			y = data_list_raw[file_index].datasets_list[rep_index].cross_list[self.channel_index - num].cross_corr_arr.y



		params = lmfit.Parameters()

		row_index = 1
		for param in self.full_dict.keys():

			params.add(param, 
				float(self.full_dict[param]["Init"].get()), 
				vary = self.fixed_list[row_index-1].get(), 
				min = float(self.full_dict[param]["Min"].get()), 
				max = float(self.full_dict[param]["Max"].get()))

			row_index+=1

 

		


		method = 'least_squares'



		o1 = lmfit.minimize(self.resid, params, args=(x, y), method=method)


		#print("# Fit using sum of squares:\n")
		#lmfit.report_fit(o1)


		output_dict = {}

		params = o1.params
		print ("Chi_Sqr = ", o1.chisqr)
		print ("Reduced Chi_Sqr = ", o1.redchi)
		popt = []
		for param in self.list_of_params:
			
			self.full_dict[param]["Init"].delete(0,"end")
			self.full_dict[param]["Init"].insert(0,str(round(params[param].value,3)))
			popt.append(np.float64(params[param].value))
			output_dict[param] = np.float64(params[param].value)






		data_list_raw[file_index].diff_fitting[rep_index, self.channel_index] = output_dict
		#print(data_list_raw[file_index].diff_fitting)

		#print(data_list_raw[file_index].diff_fitting)

		data_list_raw[file_index].diff_coeffs[rep_index, self.channel_index] = round(np.float64(self.Txy_entry.get()) * np.float64(self.D_cal_entry.get()) / params["txy"].value,3)

		if self.fit_all_flag == False:

			self.D_value.config(text = str(data_list_raw[file_index].diff_coeffs[rep_index, self.channel_index]))

		if self.channel_index < data_list_raw[file_index].datasets_list[rep_index].channels_number:

			data_list_raw[file_index].N[rep_index, self.channel_index] = round(1/params["GN0"].value,3)
			data_list_raw[file_index].cpm[rep_index, self.channel_index] = round(data_list_raw[file_index].datasets_list[rep_index].channels_list[self.channel_index].count_rate/data_list_raw[file_index].N[rep_index, self.channel_index],3)
			

			if self.fit_all_flag == False:

				self.cpm_label.config(text = str(data_list_raw[file_index].cpm[rep_index, self.channel_index]))
				self.N_label.config(text = str(data_list_raw[file_index].N[rep_index, self.channel_index]))
				
				
			





		if self.fit_all_flag == False:

			self.Plot_curve()






	def resid (self, params, x, ydata ):

		param_list = []

		for param in params.keys():

			param_list.append( np.float64(params[param].value))


		

		
		
		if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component' and self.Dimension.get() == "3D":

			y_model = Corr_curve_3d(x, *param_list)

		if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component' and self.Dimension.get() == "2D":
			y_model = Corr_curve_2d(x, *param_list)


		return y_model - ydata

	def Temp(self, event):
		print(1)

	def Update_plot(self, event):

		self.Plot_curve()


	def Plot_curve(self):

		self.save_plot_dict = {}


		global file_index
		global rep_index

		if self.fit_all_flag == False:
			self.curves.cla()

		


		num = len(data_list_raw[file_index].datasets_list[rep_index].channels_list)
		for i in range (len(data_list_raw[file_index].datasets_list[rep_index].channels_list)):

			if self.channels_flags[i].get() == 1:

				x1 = data_list_raw[file_index].datasets_list[rep_index].channels_list[i].auto_corr_arr.x
				y1 = data_list_raw[file_index].datasets_list[rep_index].channels_list[i].auto_corr_arr.y

				if self.fit_all_flag == False:
					self.curves.scatter(x1, y1, label = data_list_raw[file_index].datasets_list[rep_index].channels_list[i].short_name)

					self.save_plot_dict [data_list_raw[file_index].datasets_list[rep_index].channels_list[i].short_name] = fcs_importer.XY_plot(x1, y1)

				if 	data_list_raw[file_index].diff_fitting[rep_index, i] != None:

					popt = []

					for key in data_list_raw[file_index].diff_fitting[rep_index, i].keys():

						popt.append(np.float64(data_list_raw[file_index].diff_fitting[rep_index, i][key]))


					if len(popt) == 7:
						
						self.curves.plot(x1, Corr_curve_2d(x1, *popt), label = "Fit")

						key = str(data_list_raw[file_index].datasets_list[rep_index].channels_list[i].short_name) + " Fit"

						self.save_plot_dict [key] = fcs_importer.XY_plot(x1, Corr_curve_2d(x1, *popt))

					if len(popt) == 8:
						
						self.curves.plot(x1, Corr_curve_3d(x1, *popt), label = "Fit")

						key = str(data_list_raw[file_index].datasets_list[rep_index].channels_list[i].short_name) + " Fit"

						self.save_plot_dict [key] = fcs_importer.XY_plot(x1, Corr_curve_3d(x1, *popt))




		for i in range (len(data_list_raw[file_index].datasets_list[rep_index].cross_list)):

			if self.cross_flags[i].get() == 1:

				x1 = data_list_raw[file_index].datasets_list[rep_index].cross_list[i].cross_corr_arr.x
				y1 = data_list_raw[file_index].datasets_list[rep_index].cross_list[i].cross_corr_arr.y

				if self.fit_all_flag == False:
					self.curves.scatter(x1, y1, label = data_list_raw[file_index].datasets_list[rep_index].cross_list[i].short_name)

				k = i + len(data_list_raw[file_index].datasets_list[rep_index].channels_list)

				if 	data_list_raw[file_index].diff_fitting[rep_index, k] != None:

					popt = []


					for key in data_list_raw[file_index].diff_fitting[rep_index, k].keys():

						popt.append(np.float64(data_list_raw[file_index].diff_fitting[rep_index, k][key]))


					if len(popt) == 7:
						
						self.curves.plot(x1, Corr_curve_2d(x1, *popt), label = "Fit")

					if len(popt) == 8:
						
						self.curves.plot(x1, Corr_curve_3d(x1, *popt), label = "Fit")




		
		
		self.curves.set_title("Correlation curves")
		self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.curves.set_ylabel('G(tau)')
		self.curves.set_xlabel('Delay time')
		self.curves.set_xscale ('log')

		

		self.curves.legend(loc='upper right')

		self.canvas5.draw_idle()

		self.figure5.tight_layout()


	def Choose_curve(self, event):

		global file_index
		global rep_index
		#self.curve_index = 0

		index = self.tree.selection()
		num1, num = index[0].split('I')


		

		num = int(num, 16)

		sum1 = num 
		file = 0

		rep = 0

		ch = 0
		
		

		for i in range (len(data_list_raw)):
			#print ("I am here")
			rep = 0
			ch = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep
				ch1 = ch
			
			
			for j in range (repetitions_list[i]):
				ch = 0
				sum1-=1
				
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep
					ch1 = ch

				for k in range (total_channels_list[i]):
					sum1-=1

					ch+=1
					if sum1 == 0:
						file1 = file
						rep1 = rep
						ch1 = ch






		if rep1 == 0:
			rep1+=1

		if ch1 == 0:
			ch1+=1




		if file_index != file1-1:

			file_index = file1-1

		self.Curve_flags()
		
		rep_index = rep1-1

		self.channel_index = ch1-1



		rep = rep1-1

		self.Plot_curve()
		self.Fitting_frame()

	def Update_fitting (self, event):

		self.Fitting_frame()

	def Fitting_frame(self):

		self.frame004.destroy()

		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		if self.channel_index < data_list_raw[file_index].datasets_list[rep_index].channels_number:

			text2 = data_list_raw[file_index].datasets_list[rep_index].channels_list[self.channel_index].short_name
		else:
			imd = self.channel_index - data_list_raw[file_index].datasets_list[rep_index].channels_number
			text2 = data_list_raw[file_index].datasets_list[rep_index].cross_list[imd].short_name

		text1 = tree_list_name[file_index] + "; repetition: " + str(rep_index) + "; " + text2
		Label_1 = tk.Label(self.frame004, text=text1)
		Label_1.grid(row = 0, column = 0, columnspan = 6, sticky = 'w')

		if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component' and self.Dimension.get() == "3D" :

			self.list_of_params = ['offset', 'GN0', 'A', 'txy', 'alpha', 'AR', 'B', 'T_tri' ]
			self.list_of_inits = ['1', '1', '1', '0.02', '1', '5', '1', '0.005']
			self.list_of_min = ['0', '0', '0', '0', '0', '0', '0', '0']
			self.list_of_max = ['10', '5', '1', '100000', '20', '20', '1', '100']

		if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component' and self.Dimension.get() == "2D" :

			self.list_of_params = ['offset', 'GN0', 'A', 'txy', 'alpha', 'B', 'T_tri' ]
			self.list_of_inits = ['1', '1', '1', '0.02', '1', '1', '0.005']
			self.list_of_min = ['0', '0', '0', '0', '0',  '0', '0']
			self.list_of_max = ['10', '5', '1', '100000', '20', '1', '100']



		if 	data_list_raw[file_index].diff_fitting[rep_index, self.channel_index] != None:
			for i in range (len(self.list_of_params)):
				if self.list_of_params[i] in data_list_raw[file_index].diff_fitting[rep_index, self.channel_index].keys():
					self.list_of_inits[i] = data_list_raw[file_index].diff_fitting[rep_index, self.channel_index][self.list_of_params[i]]

		if 	data_list_raw[file_index].diff_coeffs[rep_index, self.channel_index] != None:

			diff_coef = data_list_raw[file_index].diff_coeffs[rep_index, self.channel_index]
		else:
			diff_coef = 0


		
		if self.channel_index < data_list_raw[file_index].datasets_list[rep_index].channels_number:
			if 	data_list_raw[file_index].N[rep_index, self.channel_index] != None:

				N = data_list_raw[file_index].N[rep_index, self.channel_index]
			else:
				N = 0

			if 	data_list_raw[file_index].cpm[rep_index, self.channel_index] != None:

				cpm = data_list_raw[file_index].cpm[rep_index, self.channel_index]
			else:
				cpm = 0

		#print (self.list_of_inits)

		Label_1 = tk.Label(self.frame004, text="Param")
		Label_1.grid(row = 1, column = 0, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Init")
		Label_1.grid(row = 1, column = 1, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Var")
		Label_1.grid(row = 1, column = 2, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Min")
		Label_1.grid(row = 1, column = 3, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Max")
		Label_1.grid(row = 1, column = 4, sticky = 'w')

		self.full_dict = {}
		row_index = 2

		self.fixed_list = []
		for param in self.list_of_params:
			self.fixed_list.append(tk.IntVar(value = 1))
			thisdict = {
						"Name": tk.Label(self.frame004, text=param),
						"Init": tk.Entry(self.frame004, width = 5),
						"Var": tk.Checkbutton(self.frame004, variable=self.fixed_list[row_index-2]),
						"Min": tk.Entry(self.frame004, width = 5),
						"Max": tk.Entry(self.frame004, width = 5),
						"fixed": tk.IntVar(value = 1)
						}

			self.full_dict[param] = thisdict

			thisdict["Name"].grid(row = row_index, column = 0, sticky = 'w')
			thisdict["Init"].grid(row = row_index, column = 1, sticky = 'w')
			thisdict["Init"].delete(0,"end")
			thisdict["Init"].insert(0,self.list_of_inits[row_index-2])
			thisdict["Var"].grid(row = row_index, column = 2, sticky = 'w')
			
			thisdict["Min"].grid(row = row_index, column = 3, sticky = 'w')
			thisdict["Min"].delete(0,"end")
			thisdict["Min"].insert(0,self.list_of_min[row_index-2])
			thisdict["Max"].grid(row = row_index, column = 4, sticky = 'w')
			thisdict["Max"].delete(0,"end")
			thisdict["Max"].insert(0,self.list_of_max[row_index-2])

			row_index+=1

		if self.channel_index < data_list_raw[file_index].datasets_list[rep_index].channels_number:

			self.N_label_l = tk.Label(self.frame004, text="N(FCS): ")
			self.N_label_l.grid(row = row_index, column = 0, sticky = 'w')

			self.N_label = tk.Label(self.frame004, text=str(round(N,2)))
			self.N_label.grid(row = row_index, column = 1, columnspan = 3, sticky = 'w')

			self.cpm_label_l = tk.Label(self.frame004, text="cpm (kHz): ")
			self.cpm_label_l.grid(row = row_index+1, column = 0, sticky = 'w')

			self.cpm_label = tk.Label(self.frame004, text=str(round(cpm,2)))
			self.cpm_label.grid(row = row_index+1, column = 1, columnspan = 3, sticky = 'w')

			row_index+=2

		self.D_label = tk.Label(self.frame004, text="D: ")
		self.D_label.grid(row = row_index, column = 0, sticky = 'w')

		self.D_value = tk.Label(self.frame004, text=str(round(diff_coef,2)))
		self.D_value.grid(row = row_index, column = 1, sticky = 'w')

	def Curve_flags(self):

		self.frame0003.destroy()

		self.frame0003 = tk.Frame(self.frame003)
		self.frame0003.pack(side = "top", anchor = "nw")

		self.flags_dict = {}
		self.channels_flags = []
		self.cross_flags = []

		column_counter = 0

		counter = 0

		for item in data_list_raw[file_index].datasets_list[rep_index].channels_list:
			str1, str2 = item.short_name.split(" ")
			very_short_name = "ch0" + str2
			self.channels_flags.append(tk.IntVar(value=1))
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[-1], command=self.Plot_curve)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1


		if data_list_raw[file_index].datasets_list[rep_index].cross_number > 0:

			for item in data_list_raw[file_index].datasets_list[rep_index].cross_list:
				str1, str2 = item.short_name.split(" vs ")
				str3, str4 = str1.split(" ")
				very_short_name = "ch" + str4 + str2
				self.cross_flags.append(tk.IntVar(value=1))
				self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.cross_flags[-1], command=self.Plot_curve)
				self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
				column_counter +=1



	def __init__(self, win_width, win_height, dpi_all):







		
		self.channel_index = 0
		self.fit_all_flag = False

		global file_index
		global rep_index

		global tree_list
		global tree_list_name
		global repetitions_list

		self.win_diff = tk.Toplevel()

		self.th_width = round(0.7*self.win_diff.winfo_screenwidth())
		self.th_height = round(0.4*self.win_diff.winfo_screenwidth())

		self.line1 = str(self.th_width) + "x" + str(self.th_height)


		self.win_diff.geometry(self.line1)

		self.frame002 = tk.Frame(self.win_diff)
		self.frame002.pack(side = "left", anchor = "nw")

		self.frame0002 = tk.Frame(self.frame002)
		self.frame0002.pack(side = "top", anchor = "nw")



		self.scrollbar = tk.Scrollbar(self.frame0002)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame0002, width = 100, height = 10)
		self.Datalist.pack(side = "top", anchor = "nw")
		
		
		
		self.tree = CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Choose_curve)



		self.Datalist.config(width = 100, height = 10)

		for i in range(0, len(tree_list_name)):
			name = tree_list_name[i]
			treetree = Data_tree_fcs_fit (self.tree, name, data_list_raw[i])


		self.frame003 = tk.Frame(self.frame002)
		self.frame003.pack(side = "top", anchor = "nw")

		self.frame0003 = tk.Frame(self.frame003)
		self.frame0003.pack(side = "top", anchor = "nw")



		self.frame001 = tk.Frame(self.frame002)
		self.frame001.pack(side = "top", anchor = "nw")

		self.frame000 = tk.Frame(self.win_diff)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi = dpi_all)
						
		gs = self.figure5.add_gridspec(4, 1)


		self.curves = self.figure5.add_subplot(gs[:3, 0])

		self.curves.set_title("Correlation curves")

		self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.curves.set_ylabel('G(tau)')
		self.curves.set_xlabel('Delay time (s)')

		self.residuals = self.figure5.add_subplot(gs[3, 0])

		#self.hist1.set_title("Intensity histogram")

		self.residuals.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.residuals.set_ylabel('Counts')
		self.residuals.set_xlabel('Residuals')




		self.canvas5 = FigureCanvasTkAgg(self.figure5, self.frame000)
		self.canvas5.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas5, self.frame000)
		self.toolbar.update()
		self.canvas5.get_tk_widget().pack()

		self.figure5.tight_layout()

		self.Export_plot_button = tk.Button(self.frame000, text="Save plot data", command=self.Save_plot_data)
		self.Export_plot_button.pack(side = "top", anchor = "nw")



		




		
		

		

		self.Norm_label = tk.Label(self.frame001, text="FCS curve fitting: ")
		self.Norm_label.grid(row = 0, column = 0, columnspan = 2, sticky = 'w')

		self.Triplet = ttk.Combobox(self.frame001,values = ["triplet"], width = 9 )
		self.Triplet.config(state = "readonly")
		
		self.Triplet.grid(row = 1, column = 0, sticky='ew')

		self.Triplet.set("triplet")

		self.Triplet.bind("<<ComboboxSelected>>", self.Update_fitting)

		self.Components = ttk.Combobox(self.frame001,values = ["1 component"], width = 9)
		self.Components.config(state = "readonly")
		
		self.Components.grid(row = 1, column = 1, sticky='ew')

		self.Components.set("1 component")

		self.Components.bind("<<ComboboxSelected>>", self.Update_fitting)

		self.Dimension = ttk.Combobox(self.frame001,values = ["2D", "3D"], width = 9)
		self.Dimension.config(state = "readonly")
		
		self.Dimension.grid(row = 1, column = 2, sticky='ew')

		self.Dimension.set("3D")

		self.Dimension.bind("<<ComboboxSelected>>", self.Update_fitting)



		self.Fit_button = tk.Button(self.frame001, text="Fit", command=self.Fit_corr_curve)
		self.Fit_button.grid(row = 2, column = 0, sticky='ew')



		self.Fit_all_button = tk.Button(self.frame001, text="Fit this file", command=self.Apply_to_all)
		self.Fit_all_button.grid(row = 2, column = 1, sticky='ew')

		self.Fit_button_ticked = tk.Button(self.frame001, text="Fit ticked", command=self.Apply_to_ticked)
		self.Fit_button_ticked.grid(row = 3, column = 0, sticky='ew')



		self.Fit_all_button_all = tk.Button(self.frame001, text="Fit all", command=self.Apply_to_all_all)
		self.Fit_all_button_all.grid(row = 3, column = 1, sticky='ew')

		self.Calibration_label = tk.Label(self.frame001, text="Calibration: ")
		self.Calibration_label.grid(row = 4, column = 0, columnspan = 2, sticky = 'w')

		self.D_cal_label = tk.Label(self.frame001, text="Diff coeff: ")
		self.D_cal_label.grid(row = 5, column = 0, sticky = 'w')

		self.D_cal_entry = tk.Entry(self.frame001, width = 9)
		self.D_cal_entry.grid(row = 5, column = 1, sticky='w')

		self.D_cal_entry.insert("end", str(430))

		self.Txy_label = tk.Label(self.frame001, text="Diff time: ")
		self.Txy_label.grid(row = 6, column = 0, sticky = 'w')


		self.Txy_entry = tk.Entry(self.frame001, width = 9)
		self.Txy_entry.grid(row = 6, column = 1, sticky='w')

		self.Txy_entry.insert("end", str(0.025))

		self.Table_label = tk.Label(self.frame001, text="Fitting parameters: ")
		self.Table_label.grid(row = 7, column = 0, columnspan = 2, sticky = 'w')






		
		
		



		
		self.tree.selection_set(treetree.child_id)




		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		self.Fitting_frame()
		




		#self.Plot_curve()



class Threshold_window:

	def Save_plot_data(self):

		global tree_list_name
		global file_index

		name = tree_list_name[file_index]
		filename = initialdirectory + "\\" +  name + "_Plots_gp.txt"

		open_file = open(filename, 'w')

		for key in self.save_plot_dict.keys():
			open_file.write(str(key) + "\n")

			for i in range(len(self.save_plot_dict[key].x)):
				open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

		open_file.close()

	def Apply_to_all(self):

		global rep_index

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()

		for rep_index_i in range (data_list_raw[file_index].repetitions):
			for param in self.list_of_params:
				self.full_dict[param]["Init"].delete(0,"end")
				self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))
				
			rep_index = rep_index_i
			#self.Normalize()
			self.Peaks()
			self.Fit_gaus()

		self.fit_all_flag = False

		self.Peaks()



	def Apply_to_all_ticks(self):

		global file_index
		global rep_index

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()

		global tree_list_name
		global output_file_name

		list1 = self.tree_t.get_checked()

		#print (data_frame.tree.selection())

		thisdict = {}

		for index in list1:

			num1, num = index.split('I')
			

			num = int(num, 16)

			

			sum1 = num 
			file = 0
			rep = 0
			for i in range (len(data_list_raw)):
				rep = 0
				sum1-=1
				file+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep

				
				for j in range (repetitions_list[i]):
					sum1-=1
					rep+=1
					if sum1 == 0:
						file1 = file
						rep1 = rep



			if rep1 == 0:
				rep1+=1



			file_index = file1-1
			rep_index = rep1-1




			for param in self.list_of_params:
				self.full_dict[param]["Init"].delete(0,"end")
				self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))
				
			
			#self.Normalize()
			self.Peaks()
			self.Fit_gaus()

		self.fit_all_flag = False

		self.Peaks()


	def Apply_to_all_all(self):

		global rep_index
		global file_index

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()

		for file_index_i in range (len(data_list_raw)):	
			for rep_index_i in range (data_list_raw[file_index_i].repetitions):
				for param in self.list_of_params:
					self.full_dict[param]["Init"].delete(0,"end")
					self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))
					
				rep_index = rep_index_i
				file_index = file_index_i

				#self.Normalize()
				self.Peaks()
				self.Fit_gaus()

		self.fit_all_flag = False

		self.Peaks()

	def resid (self, params, x, ydata ):

		param_list = []

		for param in self.list_of_params:

			param_list.append( np.float64(params[param].value))
		

		
		
		if self.Components.get() == '1 component':
			y_model = Gauss(x, *param_list)

		if self.Components.get() == '2 components':
			y_model = Gauss2(x, *param_list)

		if self.Components.get() == '3 components':
			y_model = Gauss3(x, *param_list)
		return y_model - ydata


	def Fit_gaus(self):

		
		

			
		
		global fit_list_x
		global fit_list_y
		global Fit_params

		x = self.x_bins
		y = self.n



		params = lmfit.Parameters()

		row_index = 1
		for param in self.list_of_params:

			params.add(param, 
				float(self.full_dict[param]["Init"].get()), 
				vary = self.fixed_list[row_index-1].get(), 
				min = float(self.full_dict[param]["Min"].get()), 
				max = float(self.full_dict[param]["Max"].get()))

			row_index+=1

 

		x1 = np.linspace(min(x), max(x), num=500)


		method = 'L-BFGS-B'

		o1 = lmfit.minimize(self.resid, params, args=(x, y), method=method)
		#print("# Fit using sum of squares:\n")
		#lmfit.report_fit(o1)


		output_dict = {}

		params = o1.params
		print ("Chi_Sqr = ", o1.chisqr)
		print ("Reduced Chi_Sqr = ", o1.redchi)
		print ("Score = ", o1.bic)
		popt = []
		for param in self.list_of_params:
			
			self.full_dict[param]["Init"].delete(0,"end")
			self.full_dict[param]["Init"].insert(0,str(round(params[param].value,3)))
			popt.append(np.float64(params[param].value))
			output_dict[param] = np.float64(params[param].value)


		data_list_raw[file_index].gp_fitting[rep_index] = output_dict

		#print(data_list_raw[file_index].gp_fitting)


			





			
		if self.fit_all_flag == False:
			self.gp_hist.cla()
										
										
			self.gp_hist.set_title("GP histogram")
			self.gp_hist.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.gp_hist.set_ylabel('Counts (Total: ' + str(sum(self.n)) + ')' )
			self.gp_hist.set_xlabel('GP')
			self.gp_hist.bar(x, y, width = x[1] - x[0], bottom=None, align='center', label = 'raw')
			

			if self.Components.get() == '1 component':
				self.gp_hist.plot(x1, Gauss(x1, *popt), 'r-', label='fit')
				self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt))

			if self.Components.get() == '2 components':
				self.gp_hist.plot(x1, Gauss2(x1, *popt), 'r-', label='fit')
				self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, Gauss2(x1, *popt))
				popt1 = popt[:3]
				popt2 = popt[3:6]
				self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')

				self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt1))
				self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt2))

			if self.Components.get() == '3 components':
				self.gp_hist.plot(x1, Gauss3(x1, *popt), 'r-', label='fit')
				self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, Gauss3(x1, *popt))
				popt1 = popt[:3]
				popt2 = popt[3:6]
				popt3 = popt[6:9]
				self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, Gauss(x1, *popt3), color = 'yellow', label='fit')

				self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt1))
				self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt2))
				self.save_plot_dict["component 3"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt3))



			self.canvas5.draw_idle()

			self.figure5.tight_layout()

	
	def Update_thresholds (self):

		global change_normal
		change_normal = False




		self.Channel_pair__choice.config(values = self.channel_pairs)
		self.Channel_pair__choice.set(self.channel_pairs[0])



		if data_list_raw[file_index].gp_fitting[rep_index] != None:
			data_list_raw[file_index].gp_fitting[rep_index] = None





		data_list_raw[file_index].threshold_list[0] = float(self.ch1_th.get())
		data_list_raw[file_index].threshold_list[1] = float(self.ch2_th.get())



			




		self.Peaks()




	def Peaks (self):

		self.save_plot_dict = {}


		if self.fit_all_flag == False:
			self.peaks.cla()
			self.hist1.cla()
			self.gp_hist.cla()

			self.canvas5.draw_idle()

			self.figure5.tight_layout()

		

		


		


		if data_list_raw[file_index].datasets_list[0].channels_number > 1:




			str1 = self.Channel_pair__choice.get()

			str7, str2 = str1.split('/')

			str3, str4 = str7.split(' ')

			str5, str6 = str2.split(' ')

			ch1_ind = int(str4) - 1

			ch2_ind = int(str6) - 1



			main_xlim = self.peaks.get_xlim()
			main_ylim = self.peaks.get_ylim()




			int_div = int(rep_index/data_list_raw[file_index].binning)

			


			x1 = []
			x2 = []
			y1 = []
			y2 = []
			y1_raw = []
			y2_raw = []


			



			for rep_index_i in range (data_list_raw[file_index].repetitions):
							
				if int(rep_index_i/data_list_raw[file_index].binning) == int_div:

					#print ("adding repetition ", rep_index_i)


					if len(x1) == 0:
						x_min = 0
					else:
						x_min = max(x1) + x1[1] - x1[0]

					x_temp_1 = [elem + x_min for elem in data_list_raw[file_index].datasets_list[rep_index_i].channels_list[ch1_ind].fluct_arr.x]
					x_temp_2 = [elem + x_min for elem in data_list_raw[file_index].datasets_list[rep_index_i].channels_list[ch2_ind].fluct_arr.x]


					x1.extend(x_temp_1)
					#y1.extend(data_list_current[file_index].datasets_list[rep_index_i].channels_list[0].fluct_arr.y)
					y1_raw.extend(data_list_raw[file_index].datasets_list[rep_index_i].channels_list[ch1_ind].fluct_arr.y)

					x2.extend(x_temp_2)
					#y2.extend(data_list_current[file_index].datasets_list[rep_index_i].channels_list[1].fluct_arr.y)
					y2_raw.extend(data_list_raw[file_index].datasets_list[rep_index_i].channels_list[ch2_ind].fluct_arr.y)




			"""if th1 == 0:
										self.ch1_th.delete(0,"end")
										self.ch1_th.insert(0,str(round(np.mean(y1),2)))
										data_list_current[file_index].threshold_ch1 = round(np.mean(y1),2)
							
									if th2 == 0:
										self.ch2_th.delete(0,"end")
										self.ch2_th.insert(0,str(round(np.mean(y2),2)))
										data_list_current[file_index].threshold_ch2 = round(np.mean(y2),2)"""



			#data_list_raw[file_index].threshold_list[0] = float(self.ch1_th.get())

			#data_list_raw[file_index].threshold_list[1] = float(self.ch2_th.get())


			#th1 = data_list_raw[file_index].threshold_list[0]
			#th2 = data_list_raw[file_index].threshold_list[1]



			th1 = data_list_raw[file_index].threshold_list[ch1_ind]
			th2 = data_list_raw[file_index].threshold_list[ch2_ind]

			if th1 == None or th2 == None:

				if self.normalization_index == "z-score":

					th1 = 2
					th2 = 2

				if self.normalization_index == "manual":


					th1 = 2
					th2 = 2

			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(th1))

				
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(th2))






			if self.normalization_index == "z-score":
				y1 = stats.zscore(y1_raw)
				y2 = stats.zscore(y2_raw)

			if self.normalization_index == "manual":


				y1 = y1_raw/np.mean(y1_raw)
				y2 = y2_raw/np.mean(y2_raw)
			

			yh1 = []
			yh2 = []



			
			

			for el in y1:
				if el >= th1:
					yh1.append(el)

			for el in y2:
				if el >= th1:
					yh2.append(el)



			which_channel = self.Threshold.get()

			
			peaks1, _ = find_peaks(y1, height=th1)

			peaks2, _ = find_peaks(y2, height=th2)



			if which_channel == "channel 1":

				peaks = peaks1

			if which_channel == "channel 2":

				peaks = peaks2

			if which_channel == "both and":

				peaks = list(set(peaks1).intersection(set(peaks2)))

			if which_channel == "both or":

				peaks = list(set(peaks1).union(set(peaks2)))

			xp1 = []
			xp2 = []
			yp1 = []

			yp1_raw = []
			yp2 = []

			yp2_raw = []







			for p in peaks:
				xp1.append(x1[p])
				xp2.append(x1[p])
				yp1.append(y1[p])
				yp2.append(y2[p])

				yp1_raw.append(y1_raw[p])
				yp2_raw.append(y2_raw[p])



			
			

			if self.fit_all_flag == False:
				self.peaks.cla()
				self.hist1.cla()
				self.gp_hist.cla()
				self.dot_plot.cla()

				self.peaks.set_title("Intensity traces")
				
				self.peaks.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
				self.peaks.set_ylabel('Intensity (a.u.)')
				self.peaks.set_xlabel('Time (s)')

				self.hist1.set_title("Intensity histograms")

				self.hist1.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
				self.hist1.set_ylabel('Counts')
				self.hist1.set_xlabel('Intensity (a.u.)')



				if which_channel == "channel 1" or which_channel == "both or" or which_channel == "both and":
					self.peaks.plot(x1, y1_raw, '#1f77b4', zorder=1)
					#self.peaks.hlines(th1, min(x1), max(x1), color = 'magenta', zorder=2)
					
					if (self.var.get() == 1):
						self.peaks.plot(xp1, yp1_raw, "x", color = 'magenta', zorder = 3)

					bins_1 = int(np.sqrt(len(yh1)))
					if bins_1 == 0:
						bins_1 = 1
					self.hist1.hist(yh1, bins = bins_1)

					#self.save_plot_dict["channel 1 fluct"] = fcs_importer.XY_plot(x1, y1)
					#self.save_plot_dict["channel 1 peaks"] = fcs_importer.XY_plot(xp1, yp1)
					

				if which_channel == "channel 2" or which_channel == "both or" or which_channel == "both and":
					
					self.peaks.plot(x2, y2_raw, '#ff7f0e', zorder=1)
					#self.peaks.hlines(th2, min(x2), max(x2), color = 'green', zorder=2)

					if (self.var.get() == 1):
						self.peaks.plot(xp2, yp2_raw, "x", color = 'green', zorder = 3)

					bins_2 = int(np.sqrt(len(yh2)))
					if bins_2 == 0:
						bins_2 = 1
					self.hist1.hist(yh2, bins = bins_2)

					#self.save_plot_dict["channel 2 fluct"] = fcs_importer.XY_plot(x2, y2)
					#self.save_plot_dict["channel 2 peaks"] = fcs_importer.XY_plot(xp2, yp2)

				"""if change_normal == False:
														self.peaks.set_xlim(main_xlim)
														self.peaks.set_ylim(main_ylim)"""

			

			gp_list_temp = []
			axis_y_temp = []
			axis_x_temp = []

			

			
			for k in range (len(yp1_raw)):
				gp_1 = (yp1_raw[k] - yp2_raw[k])/(yp2_raw[k] + yp1_raw[k])

				axis_x_temp.append(yp1_raw[k])
				axis_y_temp.append(yp2_raw[k])


				if abs(gp_1) < 1:
					gp_list_temp.append(gp_1)


			data_list_raw[file_index].datasets_list[rep_index_i].channels_list[ch1_ind].peaks = axis_x_temp
			data_list_raw[file_index].datasets_list[rep_index_i].channels_list[ch2_ind].peaks = axis_y_temp

			print(file_index, rep_index_i)
			print(data_list_raw[file_index].datasets_list[rep_index_i].channels_list[ch1_ind].peaks)
			
			self.n, bins, patches = self.gp_hist.hist(gp_list_temp, bins = int(np.sqrt(len(gp_list_temp))))

			self.dot_plot.scatter(axis_x_temp, axis_y_temp)
			self.dot_plot.ticklabel_format(axis = "x", style="sci", scilimits = (0,0))
			self.dot_plot.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))

				
			

			self.x_bins=[]
			for ii in range (len(bins)-1):
				self.x_bins.append( (bins[ii+1] - bins[ii])/2 + bins[ii])

			self.save_plot_dict["gp histogram"] = fcs_importer.XY_plot(self.x_bins, self.n)


			if self.fit_all_flag == False:
				self.gp_hist.set_title("GP histogram")
				self.gp_hist.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
				self.gp_hist.set_ylabel('Counts (Total: ' + str(len(gp_list_temp)) + ')' )
				self.gp_hist.set_xlabel('GP')


				if data_list_raw[file_index].gp_fitting[rep_index] != None:


					x1 = np.linspace(min(self.x_bins), max(self.x_bins), num=500)
					popt = []

					for param in data_list_raw[file_index].gp_fitting[rep_index].keys():
				

						popt.append(np.float64(data_list_raw[file_index].gp_fitting[rep_index][param]))




					if self.Components.get() == '1 component':
						#print("1 comp")
						self.gp_hist.plot(x1, Gauss(x1, *popt), 'r-', label='fit')

						self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt))
						print(self.save_plot_dict.keys())

					if self.Components.get() == '2 components':
						#print("2 comp")
						self.gp_hist.plot(x1, Gauss2(x1, *popt), 'r-', label='fit')
						self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, Gauss2(x1, *popt))

						popt1 = popt[:3]
						popt2 = popt[3:6]
						
						self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
						self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')

						self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt1))
						self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt2))

					if self.Components.get() == '3 components':
						self.gp_hist.plot(x1, Gauss3(x1, *popt), 'r-', label='fit')
						self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, Gauss3(x1, *popt))
						#print("3 comp")
						popt1 = popt[:3]
						popt2 = popt[3:6]
						popt3 = popt[6:9]
						
						self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
						self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')
						self.gp_hist.plot(x1, Gauss(x1, *popt3), color = 'yellow', label='fit')

						self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt1))
						self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt2))
						self.save_plot_dict["component 3"] = fcs_importer.XY_plot(x1, Gauss(x1, *popt3))





				self.canvas5.draw_idle()

				self.figure5.tight_layout()

			

	def Fitting_frame(self):


		self.frame004.destroy()



		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		if self.Components.get() == '1 component':

			self.list_of_params = ['A', 'Mean', 'Sigma' ]
			self.list_of_inits = ['0.5', '-0.16', '0.5']
			self.list_of_min = ['0', '-1', '-1']
			self.list_of_max = ['10000', '1', '1']




			

		if self.Components.get() == '2 components':

			self.list_of_params = ['A1', 'Mean1', 'Sigma1', 'A2', 'Mean2', 'Sigma2' ]
			self.list_of_inits = ['0.5', '-0.16', '0.5', '0.5', '-0.16', '0.5']
			self.list_of_min = ['0', '-1', '-1', '0', '-1', '-1']
			self.list_of_max = ['10000', '1', '1', '10000', '1', '1']

		if self.Components.get() == '3 components':

			self.list_of_params = ['A1', 'Mean1', 'Sigma1', 'A2', 'Mean2', 'Sigma2', 'A3', 'Mean3', 'Sigma3' ]
			self.list_of_inits = ['0.5', '-0.16', '0.5', '0.5', '-0.16', '0.5', '0.5', '-0.16', '0.5']
			self.list_of_min = ['0', '-1', '-1', '0', '-1', '-1', '0', '-1', '-1']
			self.list_of_max = ['10000', '1', '1', '10000', '1', '1', '10000', '1', '1']







		Label_1 = tk.Label(self.frame004, text="Param")
		Label_1.grid(row = 0, column = 0, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Init")
		Label_1.grid(row = 0, column = 1, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Var")
		Label_1.grid(row = 0, column = 2, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Min")
		Label_1.grid(row = 0, column = 3, sticky = 'w')

		Label_1 = tk.Label(self.frame004, text="Max")
		Label_1.grid(row = 0, column = 4, sticky = 'w')

		self.full_dict = {}
		row_index = 1

		self.fixed_list = []
		for param in self.list_of_params:
			self.fixed_list.append(tk.IntVar(value = 1))
			thisdict = {
						"Name": tk.Label(self.frame004, text=param),
							"Init": tk.Entry(self.frame004, width = 5),
							
							"Var": tk.Checkbutton(self.frame004, variable=self.fixed_list[row_index-1]),
							"Min": tk.Entry(self.frame004, width = 5),
							"Max": tk.Entry(self.frame004, width = 5),
						}

			self.full_dict[param] = thisdict

			thisdict["Name"].grid(row = row_index, column = 0, sticky = 'w')
			thisdict["Init"].grid(row = row_index, column = 1, sticky = 'w')
			thisdict["Init"].delete(0,"end")
			thisdict["Init"].insert(0,self.list_of_inits[row_index-1])
			thisdict["Var"].grid(row = row_index, column = 2, sticky = 'w')
			
			thisdict["Min"].grid(row = row_index, column = 3, sticky = 'w')
			thisdict["Min"].delete(0,"end")
			thisdict["Min"].insert(0,self.list_of_min[row_index-1])
			thisdict["Max"].grid(row = row_index, column = 4, sticky = 'w')
			thisdict["Max"].delete(0,"end")
			thisdict["Max"].insert(0,self.list_of_max[row_index-1])

			if data_list_raw[file_index].gp_fitting[rep_index] != None and len(data_list_raw[file_index].gp_fitting[rep_index].keys()) == len(self.list_of_params) :

				thisdict["Init"].delete(0,"end")
				thisdict["Init"].insert(0,data_list_raw[file_index].gp_fitting[rep_index][param])



			



			row_index+=1

	def Threshold_callback(self, event):

			self.Peaks()


	def Put_default(self):

		if self.normalization_index == "z-score":
			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(2))

			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(2))

		if self.normalization_index == "manual":



			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(round(np.mean(y1),2)))

			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(round(np.mean(y2),2)))

		self.Update_thresholds_button.invoke()


	def Normalize(self):

		#print ("normalize called")

		

		global file_index
		global rep_index

		
		

		#data_list_current[file_index] = copy.deepcopy(data_list_raw[file_index])

		


		

		for rep in range(repetitions_list[file_index]):


			y1 = data_list_raw[file_index].datasets_list[rep].channels_list[0].fluct_arr.y
			y2 = data_list_raw[file_index].datasets_list[rep].channels_list[1].fluct_arr.y
			
				
			if self.normalization_index == "z-score":
				y1z = stats.zscore(y1)
				y2z = stats.zscore(y2)


				data_list_raw[file_index].threshold_list[0] = float(self.ch1_th.get())
				data_list_raw[file_index].threshold_list[1] = float(self.ch2_th.get())

				#data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y = y1z
				#data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y = y2z



			


			if self.normalization_index == "manual":



				y1m = y1/np.mean(y1)
				y2m = y2/np.mean(y2)

				

				data_list_raw[file_index].threshold_list[0] = float(self.ch1_th.get())
				data_list_raw[file_index].threshold_list[1] = float(self.ch2_th.get())
				

				#data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y = y1m
				#data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y = y2m



		

		self.Peaks()






	def Normalize_index(self, event):

		self.normalization_index = self.Normalization.get()


		if self.normalization_index == "z-score":
			

			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(3))
			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(3))



		if self.normalization_index == "manual":


			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(1))
			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(1))

		if data_list_raw[file_index].gp_fitting[rep_index] != None:
			data_list_raw[file_index].gp_fitting[rep_index] = None
		
		self.Peaks()



		self.Fitting_frame()


	

	def Normalize_for_plot_index(self, event):
		
		self.normalization_index_for_plot = self.Normalization_for_plot.get()
		

	def Choose_components (self, event):

		self.Fitting_frame()

	def Plot_trace(self, event):

		global file_index
		global rep_index








		



		index = self.tree_t.selection()
		num1, num = index[0].split('I')


		

		num = int(num, 16)

		sum1 = num 
		file = 0

		rep = 0
		


		for i in range (len(data_list_raw)):
			
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1




		

		file_index = file1-1
		rep_index = rep1-1

	

		current_repetitions_number = data_list_raw[file_index].repetitions

		#print(current_repetitions_number)

		divisors = []
		for divdiv in range(1, current_repetitions_number+1):
			if current_repetitions_number % divdiv == 0:
				divisors.append(divdiv)

		self.Binning_choice.config(values = divisors)
		self.Binning_choice.set(data_list_raw[file_index].binning)


		self.channel_pairs = []
		if data_list_raw[file_index].datasets_list[0].channels_number > 1:
			for i in range (data_list_raw[file_index].datasets_list[0].channels_number):
				for j in range (i+1, data_list_raw[file_index].datasets_list[0].channels_number):
					str1 = data_list_raw[file_index].datasets_list[0].channels_list[i].short_name + "/" + data_list_raw[file_index].datasets_list[0].channels_list[j].short_name
					self.channel_pairs.append(str1)


			self.Channel_pair__choice.config(values = self.channel_pairs)
			self.Channel_pair__choice.set(self.channel_pairs[0])


		rep = rep1-1


		if data_list_raw[file_index].gp_fitting[rep_index] != None:


			if len(data_list_raw[file_index].gp_fitting[rep_index].keys()) == 3:

				

				self.Components.set("1 component")

			if len(data_list_raw[file_index].gp_fitting[rep_index].keys()) == 6:

				

				self.Components.set("2 components")

			if len(data_list_raw[file_index].gp_fitting[rep_index].keys()) == 9:

				self.Components.set("3 components")

		#self.Normalize()

		self.Peaks()

		self.Fitting_frame()



	
	def Binning(self, event):
		global file_index
		global rep_index

		global change_normal

		change_normal = True


		data_list_raw[file_index].binning = int(self.Binning_choice.get())

		

		self.Peaks()

	def __init__(self, win_width, win_height, dpi_all):


		
		self.save_plot_dict = {}

		self.fit_all_flag = False
		self.normalization_index = "z-score"

		self.normalization_index_for_plot = "raw"


		self.gp_histogram = []

		self.gp_xbins = []

		global file_index
		global rep_index

		self.win_threshold = tk.Toplevel()

		self.th_width = round(0.7*self.win_threshold.winfo_screenwidth())
		self.th_height = round(0.4*self.win_threshold.winfo_screenwidth())

		self.line1 = str(self.th_width) + "x" + str(self.th_height)


		self.win_threshold.geometry(self.line1)

		self.frame002 = tk.Frame(self.win_threshold)
		self.frame002.pack(side = "left", anchor = "nw")

		
		self.frame003 = tk.Frame(self.frame002)
		self.frame003.pack(side = "top", anchor = "nw")


		self.scrollbar_t = tk.Scrollbar(self.frame003)
		self.scrollbar_t.pack(side = "left", fill = "y")


		self.Datalist_t = tk.Listbox(self.frame003, width = 100, height = 10)
		self.Datalist_t.pack(side = "top", anchor = "nw")
		
		
		
		self.tree_t = CheckboxTreeview(self.Datalist_t)
		self.tree_t.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree_t.pack()


		self.tree_t.config(yscrollcommand = self.scrollbar_t.set)
		self.scrollbar_t.config(command = self.tree_t.yview)

		self.tree_t.bind('<<TreeviewSelect>>', self.Plot_trace)

		

		self.Datalist_t.config(width = 100, height = 10)


		self.frame001 = tk.Frame(self.frame002)
		self.frame001.pack(side = "top", anchor = "nw")


		self.frame000 = tk.Frame(self.win_threshold)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi = dpi_all)
						
		gs = self.figure5.add_gridspec(2, 3)


		self.peaks = self.figure5.add_subplot(gs[0, :3])

		self.peaks.set_title("Intensity traces")

		self.peaks.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.peaks.set_ylabel('Intensity (a.u.)')
		self.peaks.set_xlabel('Time (s)')

		self.hist1 = self.figure5.add_subplot(gs[1, 0])

		self.hist1.set_title("Intensity histogram")

		self.hist1.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.hist1.set_ylabel('Counts')
		self.hist1.set_xlabel('Intensity (a.u.)')

		self.dot_plot = self.figure5.add_subplot(gs[1, 1])

		self.dot_plot.set_title("Intensity dot plot")

		self.dot_plot.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.dot_plot.set_ylabel('channel 1')
		self.dot_plot.set_xlabel('channel 2')


		self.gp_hist = self.figure5.add_subplot(gs[1, -1])

		self.gp_hist.set_title("GP histogram")

		self.gp_hist.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.gp_hist.set_ylabel('Counts')
		self.gp_hist.set_xlabel('GP')


		self.canvas5 = FigureCanvasTkAgg(self.figure5, self.frame000)
		self.canvas5.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas5, self.frame000)
		self.toolbar.update()
		self.canvas5.get_tk_widget().pack()

		self.figure5.tight_layout()

		self.Export_plot_button = tk.Button(self.frame000, text="Save plot data", command=self.Save_plot_data)
		self.Export_plot_button.pack(side = "top", anchor = "nw")

		self.channel_pairs = []

		self.Channel_pair_label = tk.Label(self.frame001, text = "Channel pair: ")
		self.Channel_pair_label.grid(row = 0, column = 0, sticky = 'ew')

		self.Channel_pair__choice = ttk.Combobox(self.frame001,values = self.channel_pairs,  width = 18 )
		self.Channel_pair__choice.config(state = "readonly")

		self.Channel_pair__choice.grid(row = 0, column = 1, columnspan = 2, sticky = 'ew')


		self.Binning_label = tk.Label(self.frame001, text="Binning: ")
		self.Binning_label.grid(row = 1, column = 0, sticky = 'ew')

		divisors = []


		self.Binning_choice = ttk.Combobox(self.frame001,values = divisors, width = 9 )
		self.Binning_choice.config(state = "readonly")
		
		self.Binning_choice.grid(row = 1, column = 1, sticky = 'ew')

		

		self.Binning_choice.bind("<<ComboboxSelected>>", self.Binning)


		self.Norm_label = tk.Label(self.frame001, text="Use for plot: ")
		self.Norm_label.grid(row = 2, column = 0, sticky = 'ew')

		self.Normalization_for_plot = ttk.Combobox(self.frame001,values = ["raw", "normalized"], width = 9 )
		self.Normalization_for_plot.config(state = "readonly")
		
		self.Normalization_for_plot.grid(row = 2, column = 1, sticky = 'ew')

		self.Normalization_for_plot.set("raw")

		self.Normalization_for_plot.bind("<<ComboboxSelected>>", self.Normalize_for_plot_index)

		self.var = tk.IntVar()

		self.Peaks_button=tk.Checkbutton(self.frame001, text="Display peaks", variable=self.var, command=self.Update_thresholds)
		self.Peaks_button.grid(row = 2, column = 2, sticky='ew')



		




		self.Type_label = tk.Label(self.frame001, text="Detect: ")
		self.Type_label.grid(row = 3, column = 0, sticky='ew')

	

		self.Threshold = ttk.Combobox(self.frame001,values = ["both and", "both or", "channel 1", "channel 2"], width = 9 )
		self.Threshold.config(state = "readonly")
		self.Threshold.grid(row = 3, column = 1, sticky='ew')

		self.Threshold.set("both and")

		self.Threshold.bind("<<ComboboxSelected>>", self.Threshold_callback)

		
	
		
		self.Norm_label = tk.Label(self.frame001, text="Thresholding: ")
		self.Norm_label.grid(row = 4, column = 0, sticky='ew')

		self.Normalization = ttk.Combobox(self.frame001,values = ["manual", "z-score"], width = 9 )
		self.Normalization.config(state = "readonly")
								#Threshold.config(font=helv36)
		self.Normalization.grid(row = 4, column = 1, sticky = 'ew')
						
		self.Normalization.set("z-score")
						
		self.Normalization.bind("<<ComboboxSelected>>", self.Normalize_index)


		





		self.ch1_label = tk.Label(self.frame001, text="channel 1: ")
		self.ch1_label.grid(row = 5, column = 0, sticky='ew')

		self.ch1_th = tk.Entry(self.frame001, width = 9)
		self.ch1_th.grid(row = 5, column = 1, sticky='ew')

		self.ch1_th.insert("end", str(2))

		self.ch2_label = tk.Label(self.frame001, text="channel 2: ")
		self.ch2_label.grid(row = 6, column = 0, sticky='ew')

		

		self.ch2_th = tk.Entry(self.frame001, width = 9)
		self.ch2_th.grid(row = 6, column = 1, sticky='ew')

		self.ch2_th.insert("end", str(2))


		self.Update_thresholds_button = tk.Button(self.frame001, text="Update thresholds", command=self.Update_thresholds)
		self.Update_thresholds_button.grid(row = 7, column = 0, columnspan = 2, sticky='ew')

		self.Put_mean_button = tk.Button(self.frame001, text="Set to default", command=self.Put_default)
		self.Put_mean_button.grid(row = 8, column = 0, columnspan = 2, sticky='ew')







		"""self.ch1_label_zscore = tk.Label(self.frame001, text="channel 1: ")
								self.ch1_label_zscore.grid(row = 3, column = 3, sticky='w')
						
								self.ch1_th_zscore = tk.Entry(self.frame001, width = 9)
								self.ch1_th_zscore.grid(row = 3, column = 4, sticky='w')
						
								self.ch1_th_zscore.insert("end", str(data_list_current[file_index].threshold_ch1))
						
								self.ch2_label_zscore = tk.Label(self.frame001, text="channel 2: ")
								self.ch2_label_zscore.grid(row = 4, column = 3, sticky='w')
						
								
						
								self.ch2_th_zscore = tk.Entry(self.frame001, width = 9)
								self.ch2_th_zscore.grid(row = 4, column = 4, sticky='w')
						
								self.ch2_th_zscore.insert("end", str(data_list_current[file_index].threshold_ch2))
						
						
								self.Update_thresholds_button_zscore = tk.Button(self.frame001, text="Update thresholds", command=self.Peaks)
								self.Update_thresholds_button_zscore.grid(row = 5, column = 3, columnspan = 2, sticky='w')
						
								self.Put_mean_button_zscore = tk.Button(self.frame001, text="Set to mean", command=self.Put_mean)
								self.Put_mean_button_zscore.grid(row = 6, column = 3, columnspan = 2, sticky='w')"""

		#ttk.Separator(self.frame001, orient="vertical").grid(column=2, row=2, rowspan=5, sticky='ns')



		




		global tree_list
		global tree_list_name
		global repetitions_list
		


		self.frame007 = tk.Frame(self.frame002)
		self.frame007.pack(side = "top", anchor = "nw")



		self.Fit_button = tk.Button(self.frame007, text="Fit this", command=self.Fit_gaus)
		self.Fit_button.grid(row = 0, column = 0, sticky='ew')

		self.Fit_all_button = tk.Button(self.frame007, text="Fit this file", command=self.Apply_to_all)
		self.Fit_all_button.grid(row = 0, column = 1, sticky='ew')

		self.Fit_ticked_button = tk.Button(self.frame007, text="Fit ticked", command=self.Apply_to_all_ticks)
		self.Fit_ticked_button.grid(row = 1, column = 0, sticky='ew')

		self.Fit_all_all_button = tk.Button(self.frame007, text="Fit all", command=self.Apply_to_all_all)
		self.Fit_all_all_button.grid(row = 1, column = 1, sticky='ew')

		self.Components = ttk.Combobox(self.frame007,values = ["1 component", "2 components", "3 components"], width = 13 )
		self.Components.config(state = "readonly")
		self.Components.grid(row = 2, column = 0, columnspan = 2, sticky='ew')
		self.Components.set("1 component")

		self.Components.bind("<<ComboboxSelected>>", self.Choose_components)

		self.Param_label = tk.Label(self.frame007, text="Fitting parameters:")
		self.Param_label.grid(row = 3, column = 0, sticky='ew', columnspan = 2)


		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		self.Fitting_frame()


		for i in range(0, len(tree_list_name)):
			name = tree_list_name[i]
			treetree = Data_tree (self.tree_t, name, data_list_raw[i].repetitions)

		#self.Normalize()

		if self.normalization_index == "z-score":
			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(3))

			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(3))

		if self.normalization_index == "manual":



			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(round(np.mean(y1),2)))

			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(round(np.mean(y2),2)))


		self.channel_pairs = []
		if data_list_raw[file_index].datasets_list[0].channels_number > 1:
			for i in range (data_list_raw[file_index].datasets_list[0].channels_number):
				for j in range (i+1, data_list_raw[file_index].datasets_list[0].channels_number):
					str1 = data_list_raw[file_index].datasets_list[0].channels_list[i].short_name + "/" + data_list_raw[file_index].datasets_list[0].channels_list[j].short_name
					self.channel_pairs.append(str1)

		self.tree_t.selection_set(treetree.child_id)

		



	def Temp(self):
		print(1)


class Dot_Plot_Window:

	def Choose_dataset(self, event):




		index = self.tree.selection()

		num1, num = index[0].split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		


		output_file_name = tree_list_name[file1-1][:-4]
		#print(output_file_name)



		file1 = file1-1
		rep1 = rep1-1




		output_file_name = tree_list_name[file1-1][:-4]




		file_index = file1-1
		rep_index = rep1-1

		self.axis_choice = []


		

		if data_list_raw[file_index].datasets_list[0].channels_number > 1:
			for i in range (data_list_raw[file_index].datasets_list[0].channels_number):
				
				str1 = data_list_raw[file_index].datasets_list[0].channels_list[i].short_name
				self.axis_choice.append(str1)


		if data_list_raw[file_index].datasets_list[0].channels_number > 1:
			for i in range (data_list_raw[file_index].datasets_list[0].channels_number):
				
				str1 = "Diff_" + data_list_raw[file_index].datasets_list[0].channels_list[i].short_name
				self.axis_choice.append(str1)

		if data_list_raw[file_index].datasets_list[0].cross_number > 1:
			for i in range (data_list_raw[file_index].datasets_list[0].cross_number):
				
				str1 = "Diff_" + data_list_raw[file_index].datasets_list[0].cross_list[i].short_name
				self.axis_choice.append(str1)

		self.axis_choice.append("GP")


		self.Axis_y_label__choice.config(values = self.axis_choice)
		self.Axis_x_label__choice.config(values = self.axis_choice)




		




	def Plot_dataset(self):



		global file_index
		global rep_index

		self.dot_plot.cla()
		self.dens_plot.cla()

		global tree_list_name
		global output_file_name

		list1 = self.tree.get_checked()

		



		thisdict_axis_1 = {}
		thisdict_axis_2 = {}

		for index in list1:

			num1, num = index.split('I')
			

			num = int(num, 16)

			

			sum1 = num 
			file = 0
			rep = 0
			for i in range (len(data_list_raw)):
				rep = 0
				sum1-=1
				file+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep

				
				for j in range (repetitions_list[i]):
					sum1-=1
					rep+=1
					if sum1 == 0:
						file1 = file
						rep1 = rep



			if rep1 == 0:
				rep1+=1


			
			

			output_file_name = tree_list_name[file1-1][:-4]
			print("Output file name: ", output_file_name)
			print("Repetiotion: ", rep1-1)



			file1 = file1-1
			rep1 = rep1-1


			




		

			string_x = self.Axis_x_label__choice.get()
			string_y = self.Axis_y_label__choice.get()


			if string_x.__contains__("Diff") == True:

				str1, str2 = string_x.split("_")

				if data_list_raw[file1].datasets_list[rep1].channels_number > 1:
					for i in range (data_list_raw[file1].datasets_list[rep1].channels_number):
						
						if str2 == data_list_raw[file1].datasets_list[rep1].channels_list[i].short_name:
							channel_number = i

				if data_list_raw[file1].datasets_list[rep1].cross_number > 1:
					for i in range (data_list_raw[file1].datasets_list[rep1].cross_number):
						
						if str2 == data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name:
							channel_number = i + data_list_raw[file1].datasets_list[rep1].channels_number



				if output_file_name in thisdict_axis_1.keys():
					thisdict_axis_1[output_file_name].append(data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])
				else:
					thisdict_axis_1[output_file_name] = []
					thisdict_axis_1[output_file_name].append(data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])


			if string_y.__contains__("Diff") == True:

				str1, str2 = string_y.split("_")

				if data_list_raw[file1].datasets_list[rep1].channels_number > 1:
					for i in range (data_list_raw[file1].datasets_list[rep1].channels_number):
						
						if str2 == data_list_raw[file1].datasets_list[rep1].channels_list[i].short_name:
							channel_number = i



				if data_list_raw[file1].datasets_list[rep1].cross_number > 1:
					for i in range (data_list_raw[file1].datasets_list[rep1].cross_number):
						
						if str2 == data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name:
							channel_number = i + data_list_raw[file1].datasets_list[rep1].channels_number




				if output_file_name in thisdict_axis_2.keys():
					thisdict_axis_2[output_file_name].append(data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])
				else:
					thisdict_axis_2[output_file_name] = []
					thisdict_axis_2[output_file_name].append(data_list_raw[file1].diff_fitting[rep1, channel_number]["txy"])



			if string_x.__contains__("GP") == True:


				if output_file_name in thisdict_axis_1.keys():
					thisdict_axis_1[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["txy"])
				else:
					thisdict_axis_1[output_file_name] = []
					thisdict_axis_1[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["txy"])

			if string_y.__contains__("GP") == True:


				if output_file_name in thisdict_axis_2.keys():
					thisdict_axis_2[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["txy"])
				else:
					thisdict_axis_2[output_file_name] = []
					thisdict_axis_2[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["txy"])


			if string_x.__contains__("GP") == False and string_x.__contains__("Diff") == False:
				str1, str2 = string_x.split(" ")
				channel_number = int(str2) - 1


				if output_file_name in thisdict_axis_1.keys():
					thisdict_axis_1[output_file_name].append(data_list_raw[file1].datasets_list[rep1].channels_list[channel_number].peaks)
				else:
					thisdict_axis_1[output_file_name] = []
					thisdict_axis_1[output_file_name].append(data_list_raw[file1].datasets_list[rep1].channels_list[channel_number].peaks)

			if string_y.__contains__("GP") == False and string_y.__contains__("Diff") == False:
				str1, str2 = string_y.split(" ")
				channel_number = int(str2) - 1

				data_list_raw[file1].datasets_list[rep1].channels_list[channel_number].peaks

				if output_file_name in thisdict_axis_2.keys():
					thisdict_axis_2[output_file_name].append(data_list_raw[file1].datasets_list[rep1].channels_list[channel_number].peaks)
				else:
					thisdict_axis_2[output_file_name] = []
					thisdict_axis_2[output_file_name].append(data_list_raw[file1].datasets_list[rep1].channels_list[channel_number].peaks)







		
			
		"""		for key in thisdict_axis_1.keys():
			self.dot_plot.scatter(thisdict_axis_1[key], thisdict_axis_2[key], label = key )
			self.dot_plot.legend(loc='upper right')

		self.dot_plot.set_ylabel(string_x)
		self.dot_plot.set_xlabel(string_y)"""

	def __init__(self, win_width, win_height, dpi_all):

		self.channel_index = 0
		self.fit_all_flag = False

		global file_index
		global rep_index

		global tree_list
		global tree_list_name
		global repetitions_list

		self.win_dot_plot = tk.Toplevel()

		self.th_width = round(0.7*self.win_dot_plot.winfo_screenwidth())
		self.th_height = round(0.4*self.win_dot_plot.winfo_screenwidth())

		self.line1 = str(self.th_width) + "x" + str(self.th_height)


		self.win_dot_plot.geometry(self.line1)

		self.frame002 = tk.Frame(self.win_dot_plot)
		self.frame002.pack(side = "left", anchor = "nw")

		self.frame0002 = tk.Frame(self.frame002)
		self.frame0002.pack(side = "top", anchor = "nw")



		self.scrollbar = tk.Scrollbar(self.frame0002)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame0002, width = 100, height = 10)
		self.Datalist.pack(side = "top", anchor = "nw")

		
		
		
		self.tree = CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Choose_dataset)



		self.Datalist.config(width = 100, height = 10)

		for i in range(0, len(tree_list_name)):
			name = tree_list_name[i]
			treetree = Data_tree (self.tree, name, data_list_raw[i].repetitions)


		self.frame003 = tk.Frame(self.frame002)
		self.frame003.pack(side = "top", anchor = "nw")

		self.frame0003 = tk.Frame(self.frame003)
		self.frame0003.pack(side = "top", anchor = "nw")



		self.frame001 = tk.Frame(self.frame002)
		self.frame001.pack(side = "top", anchor = "nw")

		self.frame000 = tk.Frame(self.win_dot_plot)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi = dpi_all)
						
		gs = self.figure5.add_gridspec(1, 7)


		self.dot_plot = self.figure5.add_subplot(gs[0, :3])

		self.dot_plot.set_title("Dot Plot")


		self.dot_plot.set_ylabel('axis 1')
		self.dot_plot.set_xlabel('axis 2')


		self.dens_plot = self.figure5.add_subplot(gs[0, 3:6])

		self.dens_plot.set_title("Density Plot")


		self.dens_plot.set_ylabel('axis 1')
		self.dens_plot.set_xlabel('axis 2')

		self.colorbar = self.figure5.add_subplot(gs[0, 6])

		#self.hist1.set_title("Intensity histogram")

		
		



		self.canvas5 = FigureCanvasTkAgg(self.figure5, self.frame000)
		self.canvas5.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas5, self.frame000)
		self.toolbar.update()
		self.canvas5.get_tk_widget().pack()

		self.figure5.tight_layout()

		self.Export_plot_button = tk.Button(self.frame000, text="Save plot data", command=Norm)
		self.Export_plot_button.pack(side = "top", anchor = "nw")

		

		self.axis_choice = []


		self.Axis_x_label = tk.Label(self.frame001, text = "X axis: ")
		self.Axis_x_label.grid(row = 0, column = 0, sticky = 'ew')

		self.Axis_x_label__choice = ttk.Combobox(self.frame001,values = self.axis_choice,  width = 18 )
		self.Axis_x_label__choice.config(state = "readonly")

		self.Axis_x_label__choice.grid(row = 0, column = 1)

		self.Axis_y_label = tk.Label(self.frame001, text = "Y axis: ")
		self.Axis_y_label.grid(row = 1, column = 0, sticky = 'ew')

		self.Axis_y_label__choice = ttk.Combobox(self.frame001,values = self.axis_choice,  width = 18 )
		self.Axis_y_label__choice.config(state = "readonly")

		self.Axis_y_label__choice.grid(row = 1, column = 1)

		self.Plot_button = tk.Button(self.frame001, text="Plot", command=self.Plot_dataset)
		self.Plot_button.grid(row = 2, column = 0, columnspan = 2, sticky = 'ew')


		
		self.tree.selection_set(treetree.child_id)






		



	


class Data_tree:
	

	def __init__(self, tree, name, repetitions):
		

		
		
		
		self.folder1=tree.insert( "", "end", text=name)
		self.child_id = tree.get_children()[-1]
		for i in range(0, repetitions):
			text1 = "repetition " + str (i+1)
			tree.insert(self.folder1, "end", text=text1)

		#tree.focus(child_id)
		#tree.selection_set(child_id)


class Data_tree_fcs_fit:

	def __init__(self, tree, name, dataset):
		

		
		
		
		self.folder1=tree.insert( "", "end", text=name)
		self.child_id = tree.get_children()[-1]
		for i in range(0, dataset.repetitions):
			text1 = "repetition " + str (i+1)
			self.folder2=tree.insert(self.folder1, "end", text=text1)

			for j in range(dataset.datasets_list[i].channels_number):
				text1 = dataset.datasets_list[i].channels_list[j].short_name
				tree.insert(self.folder2, "end", text = text1)

			for j in range(dataset.datasets_list[i].cross_number):
				text1 = dataset.datasets_list[i].cross_list[j].short_name
				tree.insert(self.folder2, "end", text = text1)





		

			


gp_list = []

peaks_list = []

data_list_raw = []


data_list_current = []

repetitions_list = []
total_channels_list = []


root = tk.Tk()
root.title("FCS all inclusive")


screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

win_width = round(0.5 * screen_width)
win_height = round (0.8 * screen_height)

fontsize = round(win_width/85)

helv36 = tkFont.Font(size=fontsize)

line = str(win_width) + "x" + str(win_height)


root.geometry(line)

dpi_all = 75

tabs = ttk.Notebook(root, width=win_width, height=win_height, padding = 0)

tab = []

frame0 = tk.Frame(tabs)
frame1 = tk.Frame(tabs)


frame0_l = tk.LabelFrame(frame0)
frame0_l.pack(side = "left", anchor = "nw", expand = 1, fill = tk.BOTH)
frame0_l.config(bd=0, width = round(win_width * 0.5), height = win_height)
frame0_l.grid_propagate(1)

frame0_r = tk.LabelFrame(frame0)
frame0_r.pack(side = "left", anchor = "nw", expand = 1, fill = tk.BOTH)
frame0_r.config(bd=0, width = round(win_width * 0.5), height = win_height)
frame0_r.grid_propagate(1)



tabs.add(frame0, text = "Point FCS")
tabs.add(frame1, text = "Scanning FCS")

tabs_number = 2;

tabs.pack(side = "left", anchor = "nw")



data_frame = Left_frame(frame0_l, win_width, win_height, dpi_all )

#ffp = FFP_frame(frame1, win_width, win_height, dpi_all)

#diff = Diff_frame(frame0_r, win_width, win_height, dpi_all)
#gp = GP_frame(frame4, win_width, win_height, dpi_all)
#gp_diff = GP_Diff_frame(frame5, win_width, win_height, dpi_all)

root.mainloop()