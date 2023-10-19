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

import pandas as pd


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

def Corr_curve_3d_2(tc, offset, GN0, A1, A2, txy1, txy2, alpha1, alpha2, AR1, AR2, B1, tauT1 ):

	txy1 = txy1 / 1000

	tauT1 = tauT1 / 1000

	G_Diff =  (A1*(((1+((tc/txy1)**alpha1))**-1)*(((1+(tc/((AR1**2)*txy1)))**-0.5)))) + (A2*(((1+((tc/txy2)**alpha2))**-1)*(((1+(tc/((AR2**2)*txy2)))**-0.5))))

	G_T = 1 + (B1*np.exp(tc/(-tauT1)))

	return offset + GN0 * G_Diff * G_T

def Corr_curve_2d_2(tc, offset, GN0, A1, A2, txy1, txy2, alpha1, alpha2, B1, tauT1):

	txy1 = txy1 / 1000

	tauT1 = tauT1 / 1000

	G_Diff =  A1*(((1+((tc/txy1)**alpha1))**-1)) + A2*(((1+((tc/txy2)**alpha2))**-1))

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

	ffp.main.cla()
	#ffp.figure3.clf()

	



	list1 = data_c.data_frame.tree.get_checked()

	flag = False

	x1 = []
	y1 = []

	for index in list1:
		num1, num = index.split('I')
		

		num = int(num, 16)

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_c.data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (data_c.repetitions_list[i]):
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



		all_bins_x = np.logspace(np.log10(min_x1),np.log10(np.max(x1)), num=bins_number)

		#all_bins_x = np.logspace(np.log10(min(x1)), np.log10(max(x1)), num=int(np.sqrt(bins_number)),  base=10.0)
		#all_bins_y = np.logspace(np.log10(min(y1)), np.log10(max(y1)), num=int(np.sqrt(bins_number)),  base=10.0)


	

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
	data_c.data_frame.gp_plot.cla()



	list1 = data_c.data_frame.tree.get_checked()

	#print(list1)

	#print (data_frame.tree.selection())

	thisdict = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_c.data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (data_c.repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		
		#print(rep1)

		data_c.output_file_name = data_c.tree_list_name[file1-1][:-4]
		#print(output_file_name)



		file1 = file1-1
		rep1 = rep1-1




		if data_c.data_list_raw[file1].gp_fitting[rep1] != None:

			if len(data_c.data_list_raw[file1].gp_fitting[rep1].keys()) == 4:


				if data_c.output_file_name in thisdict.keys():

					thisdict[data_c.output_file_name].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean"])
				else:
					thisdict[data_c.output_file_name] = []
					thisdict[data_c.output_file_name].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean"])

			if len(data_c.data_list_raw[file1].gp_fitting[rep1].keys()) == 7:

				key = data_c.output_file_name + " peak 1"


				if key in thisdict.keys():

					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean1"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean1"])

				key = data_c.output_file_name + " peak 2"


				if key in thisdict.keys():

					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean2"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean2"])


			if len(data_c.data_list_raw[file1].gp_fitting[rep1].keys()) == 10:

				key = data_c.output_file_name + " peak 1"


				if key in thisdict.keys():

					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean1"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean1"])

				key = data_c.output_file_name + " peak 2"


				if key in thisdict.keys():

					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean2"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean2"])

				key = data_c.output_file_name + " peak 3"


				if key in thisdict.keys():

					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean3"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean3"])

		
		
		


	keys = []
	vals = []
	for key in thisdict.keys():
		keys.append(key)
		vals.append(thisdict[key])


	if vals:
		#sns.axlabel( ylabel="GP", fontsize=16)
		sns.boxplot(data=vals, width=.18, ax = data_c.data_frame.gp_plot)
		#sns.swarmplot(data=vals, ax = data_c.data_frame.gp_plot)
		sns.stripplot( data=vals, ax = data_c.data_frame.gp_plot, jitter = True)

		# category labels
		data_c.data_frame.gp_plot.set_xticklabels(keys, rotation = 45)
		#diff.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		data_c.data_frame.gp_plot.set_ylabel('GP')
		#data_c.data_frame.diff_plot.xticks(rotation = 45)


	
	
def Plot_diff():
	diff_list = []
	data_c.data_frame.diff_plot.cla()



	list1 = data_c.data_frame.tree.get_checked()



	thisdict = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_c.data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (data_c.repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		


		data_c.output_file_name = data_c.tree_list_name[file1-1][:-4]





		file1 = file1-1
		rep1 = rep1-1


		"""if output_file_name in thisdict.keys():
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])
								else:
									thisdict[output_file_name] = []
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])"""

		for item in range(len(data_c.data_list_raw[file1].datasets_list[rep1].channels_list)):

			if data_c.data_frame.channels_flags[data_c.data_list_raw[file1].datasets_list[rep1].channels_list[item].short_name].get() == 1 and data_c.data_list_raw[file1].diff_fitting[rep1, item]!= None:

				key = data_c.output_file_name + " " + data_c.data_list_raw[file1].datasets_list[rep1].channels_list[item].short_name 

				if key in thisdict.keys():
					thisdict[key].append(data_c.data_list_raw[file1].diff_fitting[rep1, item]["txy"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_c.data_list_raw[file1].diff_fitting[rep1, item]["txy"])

		for i in range(len(data_c.data_list_raw[file1].datasets_list[rep1].cross_list)):

			item = i + len(data_c.data_list_raw[file1].datasets_list[rep1].channels_list)



			if data_c.data_frame.channels_flags[data_c.data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name].get() == 1 and data_c.data_list_raw[file1].diff_fitting[rep1, item] != None:

				key = data_c.output_file_name + " " + data_c.data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name 

				if key in thisdict.keys():
					thisdict[key].append(data_c.data_list_raw[file1].diff_fitting[rep1, item]["txy"])
				else:
					thisdict[key] = []
					thisdict[key].append(data_c.data_list_raw[file1].diff_fitting[rep1, item]["txy"])

		



	keys = []
	vals = []
	for key in thisdict.keys():
		keys.append(key)
		vals.append(thisdict[key])


	if vals:
		sns.set(context='notebook', style='whitegrid')
		#sns.axlabel( ylabel="Diffusion time", fontsize=16)
		sns.boxplot(data=vals, width=.18, ax = data_c.data_frame.diff_plot)
		#sns.swarmplot(data=vals,  ax = data_c.data_frame.diff_plot)
		sns.stripplot( data=vals, ax = data_c.data_frame.gp_plot, jitter = True)

		# category labels
		data_c.data_frame.diff_plot.set_xticklabels(keys, rotation = 45)
		#diff.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		data_c.data_frame.diff_plot.set_ylabel('Diffusion time')
		#data_c.data_frame.diff_plot.xticks(rotation = 45)
		

def Plot_gp_diff():

	gp_diff.main.cla()



	list1 = data_c.data_frame.tree.get_checked()



	thisdict_gp = {}
	thisdict_diff = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_c.data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (data_c.repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		


		data_c.output_file_name = data_c.tree_list_name[file1-1][:-4]




		file1 = file1-1
		rep1 = rep1-1


		"""if output_file_name in thisdict.keys():
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])
								else:
									thisdict[output_file_name] = []
									thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])"""

		if data_c.output_file_name in thisdict_diff.keys():
			thisdict_diff[data_c.output_file_name].append(data_c.data_list_raw[file1].diff_fitting[rep1, 0]["txy"])
		else:
			thisdict_diff[data_c.output_file_name] = []
			thisdict_diff[data_c.output_file_name].append(data_c.data_list_raw[file1].diff_fitting[rep1, 0]["txy"])


		if data_c.output_file_name in thisdict_gp.keys():
			thisdict_gp[data_c.output_file_name].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean"])
		else:
			thisdict_gp[data_c.output_file_name] = []
			thisdict_gp[data_c.output_file_name].append(data_c.data_list_raw[file1].gp_fitting[rep1]["Mean"])

		
	for key in thisdict_diff.keys():
		gp_diff.main.scatter(thisdict_gp[key], thisdict_diff[key], label = key )
		gp_diff.main.legend(loc='upper right')

	gp_diff.main.set_ylabel('Diffusion time (ms)')
	gp_diff.main.set_xlabel('GP')






def Which_tab():



	Plot_diff()


	Plot_gp()

	data_c.data_frame.canvas1.draw()
	data_c.data_frame.figure1.tight_layout()



	#if tabs.index(tabs.select()) == 2:
		#Plot_gp_diff()

		#gp_diff.canvas3.draw()

		#gp_diff.figure3.tight_layout()

	#except:
		#tk.messagebox.showerror(title='Error', message=Message_generator())



def Threshold_fun():

	print(data_c.initialdirectory)


	if len(data_c.tree_list_name) > 0:

		th_win = gp_win.Threshold_window(data_c.win_width, data_c.win_height, data_c.dpi_all)

	if len(data_c.tree_list_name) == 0:

		tk.messagebox.showerror(title='Error', message=Message_generator())

def Diffusion_fun():
	if len(data_c.tree_list_name) > 0:

		th_win = diff_win.Diffusion_window(data_c.win_width, data_c.win_height, data_c.dpi_all)

	if len(data_c.tree_list_name) == 0:

		tk.messagebox.showerror(title='Error', message=Message_generator())

def Dot_Plot_fun():

	if len(data_c.tree_list_name) > 0:

		dot_plot_win = dot_plot.Dot_Plot_Window(data_c.win_width, data_c.win_height, data_c.dpi_all)

	if len(data_c.tree_list_name) == 0:

		tk.messagebox.showerror(title='Error', message=Message_generator())


def Restruct_fun():


	th_win = restruct_d.Restruct_window(data_c.win_width, data_c.win_height, data_c.dpi_all)


def Export_function():




	now = datetime.now()
	str1, str2 = str(now).split(".")
	name_dir = str1 + " Analysis"

	name_dir = name_dir.replace(":", "_")

	

	directory = os.path.join(data_c.initialdirectory, name_dir) 
    

	os.mkdir(directory) 




	data_c.output_numbers_dict = {}





	list1 = data_c.data_frame.tree.get_checked()





	thisdict_gp = {}
	thisdict_diff = {}

	for index in list1:

		num1, num = index.split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_c.data_list_raw)):
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (data_c.repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1


		file1 = file1-1
		rep1 = rep1-1

		if file1 in data_c.output_numbers_dict.keys():
			data_c.output_numbers_dict[file1].append(rep1)
		else:
			data_c.output_numbers_dict[file1] = []
			data_c.output_numbers_dict[file1].append(rep1)

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

	for file1 in data_c.output_numbers_dict.keys():

		heading += str(file1)

		heading_line_1 += str(file1) + "\t\t\t"

		heading_line_2 += "GP" + "\t" + "Diffusion_coef" + "\t" + "Diffusion_time" + "\t"

		data_c.output_file_name, str2 = data_c.tree_list_name[file1].split(".")



		filename = directory + os.path.sep + data_c.output_file_name + ".xlsx"

		open_file = open (filename, "w")

		writer = pd.ExcelWriter(filename, engine='xlsxwriter')

		#open_file.write(data_c.output_file_name + "\n")

		#open_file.write("Diffusion data: \n")

		chan = {}


		for channel in range(len(data_c.data_list_raw[file1].datasets_list[rep1].channels_list)):

			if data_c.data_frame.channels_flags[data_c.data_list_raw[file1].datasets_list[rep1].channels_list[channel].short_name].get() == 1:

				chan[data_c.data_list_raw[file1].datasets_list[rep1].channels_list[channel].short_name] = channel


		for i in range (len(data_c.data_list_raw[file1].datasets_list[rep1].cross_list)):

			channel = i + len(data_c.data_list_raw[file1].datasets_list[rep1].channels_list)

			if data_c.data_frame.channels_flags[data_c.data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name].get() == 1:

				chan[data_c.data_list_raw[file1].datasets_list[rep1].cross_list[i].short_name] = channel

		
		"""for name in chan.keys():
						
									channel = chan[name]
									open_file.write(name + "\n")
						
									line = "name\t"
						
									rep0 = data_c.output_numbers_dict[file1][0]
						
									if data_c.data_list_raw[file1].diff_fitting[rep0, channel] != None:
						
										for key in data_c.data_list_raw[file1].diff_fitting[rep0, channel].keys():
						
											line += key + "\t"
						
										line += "N" + "\t"
										line += "cpm" + "\t"
										line += "D" + "\t"
						
						
									open_file.write(line + "\n")"""



		for name in chan.keys():

			try:

				channel = chan[name]

				dict_temp_list =[]
				for rep1 in data_c.output_numbers_dict[file1]:

					dict_temp = {}				

					for key in data_c.data_list_raw[file1].diff_fitting[rep1, channel].keys():


						dict_temp[key] = data_c.data_list_raw[file1].diff_fitting[rep1, channel][key]


					counter = 1
					
					for item in data_c.data_list_raw[file1].diff_coeffs[rep1, channel]:

						key = "D_" + str(counter)

						dict_temp[key] = item

						counter+=1


					try:
						dict_temp["N"] = data_c.data_list_raw[file1].N[rep1, channel]

						dict_temp["cpm"] = data_c.data_list_raw[file1].cpm[rep1, channel]

					except:
						pass


					dict_temp_list.append(dict_temp)


				df1 = pd.DataFrame.from_records(dict_temp_list)


				df1.to_excel(writer, sheet_name=name)

			except:
				pass





		#open_file.write("\n")
		#open_file.write("GP data: \n")

		#rep0 = data_c.output_numbers_dict[file1][0]

		#line = "name\t"

		try:

			dict_temp_list = []

			for rep1 in data_c.output_numbers_dict[file1]:

				dict_temp = {}
			

				for key in data_c.data_list_raw[file1].gp_fitting[rep1].keys():
					

					dict_temp[key] = data_c.data_list_raw[file1].gp_fitting[rep1][key]
					#line += key + "\t"

				dict_temp_list.append(dict_temp)

			df1 = pd.DataFrame.from_records(dict_temp_list)


			df1.to_excel(writer, sheet_name="GP")

		except:
			pass

		#open_file.write(line + "\n")

		"""for rep1 in data_c.output_numbers_dict[file1]:
						
									line = "Repetition " + str(rep1 + 1) + "\t"
						
									try:
						
										for key in data_c.data_list_raw[file1].gp_fitting[rep1].keys():
						
											line += str(data_c.data_list_raw[file1].gp_fitting[rep1][key]) + "\t"
						
									except:
										pass
						
						
									open_file.write(line + "\n")"""





		writer.close()


	filename = directory + os.path.sep + "Summary.xlsx"

	df_gp = pd.DataFrame ()

	df_sigma = pd.DataFrame ()

	df_totals = pd.DataFrame ()

	df_diff_coeff = {}

	df_diff_time = {}

	df_cpms = {}

	df_T = pd.DataFrame ()

	df_diff_time_list = {}

	df_cpm_list = {}

	df_T_list = {}

	for file1 in data_c.output_numbers_dict.keys():
		heading_line = str(data_c.tree_list_name[file1])

		list_gp_mean = []
		list_gp_sigma = []
		list_gp_totals = []

		list_diff_coeffs = {}
		list_diff_times = {}
		list_cpms = {}

		for el in data_c.data_list_raw[file1].gp_fitting:
			try:
				list_gp_mean.append(el["Mean"])
				list_gp_sigma.append(el["Sigma"])
				list_gp_totals.append(el["Total peaks"])

			except:
				pass





		
		for chan in range(0, data_c.data_list_raw[file1].datasets_list[0].channels_number):

			#try:

			


			if data_c.data_list_raw[file1].diff_fitting[0, chan] != None:
				list_T_temp = []
				list_D_temp = []
				list_C_temp = []
				if "txy" in data_c.data_list_raw[file1].diff_fitting[0, chan].keys():
					for rep in range(0, data_c.repetitions_list[file1]):

					
					
						list_T_temp.append(data_c.data_list_raw[file1].diff_fitting[rep, chan]["txy"])
						list_D_temp.append(data_c.data_list_raw[file1].diff_coeffs[rep, chan][0])
						list_C_temp.append(data_c.data_list_raw[file1].cpm[rep, chan])

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "txy"
					list_diff_times [key] = copy.deepcopy(list_T_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "D"
					list_diff_coeffs [key] = copy.deepcopy(list_D_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "cpm"
					list_cpms [key] = copy.deepcopy(list_C_temp)

				list_T_temp = []
				list_D_temp = []
				list_C_temp = []

				if "txy1" in data_c.data_list_raw[file1].diff_fitting[0, chan].keys():
					for rep in range(0, data_c.repetitions_list[file1]):

					
					
						list_T_temp.append(data_c.data_list_raw[file1].diff_fitting[rep, chan]["txy1"])
						list_D_temp.append(data_c.data_list_raw[file1].diff_coeffs[rep, chan][0])
						list_C_temp.append(data_c.data_list_raw[file1].cpm[rep, chan])

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "txy1"
					list_diff_times [key] = copy.deepcopy(list_T_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "D"
					list_diff_coeffs [key] = copy.deepcopy(list_D_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" +  "cpm"
					list_cpms [key] = copy.deepcopy(list_C_temp)


				list_T_temp = []
				list_D_temp = []
				list_C_temp = []


				if "txy2" in data_c.data_list_raw[file1].diff_fitting[0, chan].keys():
					for rep in range(0, data_c.repetitions_list[file1]):

					
					
						list_T_temp.append(data_c.data_list_raw[file1].diff_fitting[rep, chan]["txy2"])
						list_D_temp.append(data_c.data_list_raw[file1].diff_coeffs[rep, chan][1])
						list_C_temp.append(data_c.data_list_raw[file1].cpm[rep, chan])

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "txy2"
					list_diff_times [key] = copy.deepcopy(list_T_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "D"
					list_diff_coeffs [key] = copy.deepcopy(list_D_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "cpm"
					list_cpms [key] = copy.deepcopy(list_C_temp)

				list_T_temp = []
				list_D_temp = []
				list_C_temp = []


				if "txy3" in data_c.data_list_raw[file1].diff_fitting[0, chan].keys():
					for rep in range(0, data_c.repetitions_list[file1]):

					
					
						list_T_temp.append(data_c.data_list_raw[file1].diff_fitting[rep, chan]["txy3"])
						list_D_temp.append(data_c.data_list_raw[file1].diff_coeffs[rep, chan][2])
						list_C_temp.append(data_c.data_list_raw[file1].cpm[rep, chan])

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "txy3"
					list_diff_times [key] = copy.deepcopy(list_T_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "D"
					list_diff_coeffs [key] = copy.deepcopy(list_D_temp)

					key = data_c.data_list_raw[file1].datasets_list[rep].channels_list[chan].short_name + "_" + "cpm"
					list_cpms [key] = copy.deepcopy(list_C_temp)




			#except:
				#print("Something went wrong")






		df1_gp = pd.DataFrame({heading_line: list_gp_mean})
		df_gp = pd.concat([df_gp, df1_gp], axis=1)

		df1_sigma = pd.DataFrame({heading_line: list_gp_sigma})
		df_sigma = pd.concat([df_sigma, df1_sigma], axis=1)

		df1_totals = pd.DataFrame({heading_line: list_gp_totals})
		df_totals = pd.concat([df_totals, df1_totals], axis=1)



		"""for chan in range(0, data_c.data_list_raw[file1].datasets_list[0].channels_number):

			df1_T = pd.DataFrame({heading_line: list_diff_times[chan]})
			df1_D = pd.DataFrame({heading_line: list_diff_coeffs[chan]})
			df1_C = pd.DataFrame({heading_line: list_cpms[chan]})


			if chan in df_diff_time:
				df_diff_time [chan] = pd.concat([df_diff_time [chan], df1_T], axis=1)
				df_diff_coeff [chan] = pd.concat([df_diff_coeff[chan], df1_D], axis=1)
				df_cpms [chan] = pd.concat([df_cpms[chan], df1_C], axis=1)
			
			else:
				df_diff_time [chan] = pd.DataFrame ()
				df_diff_coeff [chan] = pd.DataFrame ()
				df_cpms [chan] = pd.DataFrame ()

				df_diff_time [chan] = pd.concat([df_diff_time [chan], df1_T], axis=1)
				df_diff_coeff [chan] = pd.concat([df_diff_coeff[chan], df1_D], axis=1)
				df_cpms [chan] = pd.concat([df_cpms[chan], df1_C], axis=1)"""

		for chan in list_diff_times:
			df1_T = pd.DataFrame({heading_line: list_diff_times[chan]})

			if chan in df_diff_time:
				df_diff_time [chan] = pd.concat([df_diff_time [chan], df1_T], axis=1)

			else:
				df_diff_time [chan] = pd.DataFrame ()
				df_diff_time [chan] = pd.concat([df_diff_time [chan], df1_T], axis=1)

		for chan in list_diff_coeffs:

			df1_D = pd.DataFrame({heading_line: list_diff_coeffs[chan]})

			if chan in df_diff_coeff:
				df_diff_coeff [chan] = pd.concat([df_diff_coeff[chan], df1_D], axis=1)
			else:
				df_diff_coeff [chan] = pd.DataFrame ()
				df_diff_coeff [chan] = pd.concat([df_diff_coeff[chan], df1_D], axis=1)

		for chan in list_cpms:

			df1_C = pd.DataFrame({heading_line: list_cpms[chan]})

			if chan in df_cpms:
				df_cpms [chan] = pd.concat([df_cpms[chan], df1_C], axis=1)
			else:
				df_cpms [chan] = pd.DataFrame ()
				df_cpms [chan] = pd.concat([df_cpms[chan], df1_C], axis=1)







		


			

		





	writer = pd.ExcelWriter(filename, engine='xlsxwriter')

	df_gp.to_excel(writer, sheet_name='GP_Mean')
	df_sigma.to_excel(writer, sheet_name='GP_Sigma')
	df_totals.to_excel(writer, sheet_name='Total peaks')

	for key in df_diff_time:
		legend = key
		df_diff_time [key].to_excel(writer, sheet_name=legend)
		#print(df_diff_time [key])

	for key in df_diff_coeff:
		legend = key
		df_diff_coeff [key].to_excel(writer, sheet_name=legend)

	for key in df_cpms:
		legend = key
		df_cpms [key].to_excel(writer, sheet_name=legend)


	writer.close()