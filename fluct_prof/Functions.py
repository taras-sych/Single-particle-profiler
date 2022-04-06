#--------------------------
#Importing general modules
#--------------------------
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


#--------------------------
#End of importing general modules
#--------------------------



#--------------------------
#Importing own modules
#--------------------------


from fluct_prof import Restructure_data as restr_d

from fluct_prof import Diffusion_window as diff_win

from fluct_prof import GP_Window as gp_win

from fluct_prof import Dot_Plot as dot_plot

from fluct_prof import Restructure_data as restruct_d

from fluct_prof import Data_container as data_c


#--------------------------
#End of importing own modules
#--------------------------






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

	if len(data_c.tree_list_name) > 0:

		th_win = gp_win.Threshold_window(data_c.win_width, data_c.win_height, data_c.dpi_all)

	if len(data_c.tree_list_name) == 0:

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


	th_win = restruct_d.Restruct_window(data_c.win_width, data_c.win_height, data_c.dpi_all)


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