import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import matplotlib.pyplot as plt

import lmfit


from pandastable import Table
from pandas import DataFrame

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2TkAgg)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib import cm as mplcm

from ttkwidgets import CheckboxTreeview

import fcs_importer

import Correlation as corr_py

import codecs

import os

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

def Message_generator():
	messages = [
    'You shall not pass!',  
    'Danger!',
    'She is dead, Jim!',
    'My life for the horde!' 
	] 
	index = random.randint(0, len(messages)-1)
	return messages[index]

def Corr_curve(tc, offset, GN0, A1, txy1, alpha1, AR1, B1, tauT1 ):

	txy1 = txy1 / 1000

	tauT1 = tauT1 / 1000

	G_Diff =  (A1*(((1+((tc/txy1)**alpha1))**-1)*(((1+(tc/((AR1**2)*txy1)))**-0.5))))

	G_T = 1 + (B1*np.exp(-tc/tauT1))

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
	gp.main.cla()

	global tree_list_name
	global output_file_name

	list1 = data_frame.tree.get_checked()

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


		


		output_file_name = tree_list_name[file1-1][:-4]
		#print(output_file_name)



		file1 = file1-1
		rep1 = rep1-1


		if output_file_name in thisdict.keys():
			thisdict[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["Mean"])
		else:
			thisdict[output_file_name] = []
			thisdict[output_file_name].append(data_list_raw[file1].gp_fitting[rep1]["Mean"])

		
		
		


	keys = []
	vals = []
	for key in thisdict.keys():
		keys.append(key)
		vals.append(thisdict[key])



	
	#sns.axlabel( ylabel="Diffusion time", fontsize=16)
	sns.boxplot(data=vals, width=.18, ax = gp.main)
	sns.swarmplot(data=vals, size=6, edgecolor="black", linewidth=.9, ax = gp.main)

	# category labels
	gp.main.set_xticklabels(keys)
	#diff.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))


	
	
def Plot_diff():
	diff_list = []
	diff.main.cla()

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

		if output_file_name in thisdict.keys():
			thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])
		else:
			thisdict[output_file_name] = []
			thisdict[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])

		

		



	keys = []
	vals = []
	for key in thisdict.keys():
		keys.append(key)
		vals.append(thisdict[key])



	sns.set(context='notebook', style='whitegrid')
	#sns.axlabel( ylabel="Diffusion time", fontsize=16)
	sns.boxplot(data=vals, width=.18, ax = diff.main)
	sns.swarmplot(data=vals, size=6, edgecolor="black", linewidth=.9, ax = diff.main)

	# category labels
	diff.main.set_xticklabels(keys)
	#diff.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		

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
			thisdict_diff[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])
		else:
			thisdict_diff[output_file_name] = []
			thisdict_diff[output_file_name].append(data_list_raw[file1].diff_fitting[rep1]["txy"])


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


	try:
		if tabs.index(tabs.select()) == 0:
			Plot_diff()

			diff.canvas3.draw()

			diff.figure3.tight_layout()
			

		if tabs.index(tabs.select()) == 1:
			Plot_gp()

			gp.canvas3.draw()

			gp.figure3.tight_layout()

		if tabs.index(tabs.select()) == 2:
			Plot_gp_diff()

			gp_diff.canvas3.draw()

			gp_diff.figure3.tight_layout()

	except:
		tk.messagebox.showerror(title='Error', message=Message_generator())



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


def Restruct_fun():


	th_win = Restruct_window(win_width, win_height, dpi_all)

class Restruct_window:

	def Temp(self):
		print(1)



	def Plot_curve(self):


		global file_index
		global rep_index

		
		self.curves.cla()
		self.traces.cla()





		x1 = data_list_raw[file_index].datasets_list[rep_index].channels_list[0].auto_corr_arr.x
		y1 = data_list_raw[file_index].datasets_list[rep_index].channels_list[0].auto_corr_arr.y

		self.x_fit = x1
		self.y_fit = y1

		
		self.curves.plot(x1, y1, label = "auto corr ch 1")


		


		x2 = data_list_raw[file_index].datasets_list[rep_index].channels_list[1].auto_corr_arr.x
		y2 = data_list_raw[file_index].datasets_list[rep_index].channels_list[1].auto_corr_arr.y

		
		self.curves.plot(x2, y2, label = "auto corr ch 2")




		x3 = data_list_raw[file_index].datasets_list[rep_index].cross_list[0].cross_corr_arr.x
		y3 = data_list_raw[file_index].datasets_list[rep_index].cross_list[0].cross_corr_arr.y

		
		self.curves.plot(x3, y3, label = "cross-corr")

		
		
		self.curves.set_title("Correlation curves")
		self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.curves.set_ylabel('G(tau)')
		self.curves.set_xlabel('Delay time')
		self.curves.set_xscale ('log')


		y1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.y
		y2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.y

		x1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.x
		x2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.x


		self.traces.set_title("Intensity traces")
		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('Intensity')
		self.traces.set_xlabel('Time (s)')

		self.traces.plot(x1, y1, label = "channel 1")
		self.traces.plot(x2, y2, label = "channel 2")

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
		self.num_rep.insert(0,data_list_current[rep_index].repetitions)

		self.Rep_button = tk.Button(self.frame004, text="Apply reps", command=self.Temp)
		self.Rep_button.grid(row = 0, column = 2, sticky='w')

		self.Remove_button = tk.Button(self.frame004, text="Remove dataset", command=self.Temp)
		self.Remove_button.grid(row = 1, column = 0, columnspan = 3, sticky = 'ew')

		self.frame000 = tk.Frame(self.win_diff)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi=100)
						
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
			Data_tree (self.tree, name, data_list_current[i].repetitions)





class Left_frame :


	def Normalize(self):
		global file_index
		global rep_index


		for rep in range (repetitions_list[file_index]):
			y1 = data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y
			y2 = data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y
		
			if self.normalization_index == "z-score":

				y1z = stats.zscore(y1)
				y2z = stats.zscore(y2)

				data_list_current[file_index].threshold_ch1 = 1
				data_list_current[file_index].threshold_ch2 = 1

				self.Plot_this_data(data_list_current[file_index], rep_index)


			if self.normalization_index == "mean":

				y1 = y1/np.mean(y1)
				y2 = y2/np.mean(y2)

				data_list_current[file_index] = copy.deepcopy(data_list_raw[file_index])

				data_list_current[file_index].threshold_ch1 = 0
				data_list_current[file_index].threshold_ch2 = 0

				self.Plot_this_data(data_list_current[file_index], rep_index)






		#data_list_current[file_index].threshold_ch1 = 0
		#data_list_current[file_index].threshold_ch2 = 0





		#dataset = data_list[file_index]

		self.Plot_this_data(data_list_current[file_index], rep_index)

	def Restore(self):
		global file_index
		global rep_index

		#print(1)

		data_list_current[file_index] = copy.deepcopy(data_list_raw[file_index])

		data_list_current[file_index].threshold_ch1 = 0
		data_list_current[file_index].threshold_ch2 = 0

		self.Plot_this_data(data_list_current[file_index], rep_index)


	def Plot_this_data(self, datasets_pos, rep):
		x1 = datasets_pos.datasets_list[rep].channels_list[0].fluct_arr.x
		y1 = datasets_pos.datasets_list[rep].channels_list[0].fluct_arr.y

		x2 = datasets_pos.datasets_list[rep].channels_list[1].fluct_arr.x
		y2 = datasets_pos.datasets_list[rep].channels_list[1].fluct_arr.y


		self.traces.cla()

		self.traces.set_title("Intensity traces")
		
		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('Intensity (a.u.)')
		self.traces.set_xlabel('Time (s)')

		self.traces.plot(x1, y1, label = "channel 1")
		self.traces.plot(x2, y2, label = "channel 2")

		self.traces.legend(loc='upper right')




		self.corr.set_title("Correlation curves")


		self.corr.cla()

		x1 = datasets_pos.datasets_list[rep].channels_list[0].auto_corr_arr.x
		y1 = datasets_pos.datasets_list[rep].channels_list[0].auto_corr_arr.y

		x2 = datasets_pos.datasets_list[rep].channels_list[1].auto_corr_arr.x
		y2 = datasets_pos.datasets_list[rep].channels_list[1].auto_corr_arr.y

		x3 = datasets_pos.datasets_list[rep].cross_list[0].cross_corr_arr.x
		y3 = datasets_pos.datasets_list[rep].cross_list[0].cross_corr_arr.y

		self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.corr.set_ylabel('G(tau)')
		self.corr.set_xlabel('Delay time')
		self.corr.set_xscale ('log')

		self.corr.plot(x1, y1, label = "auto corr ch 1")
		self.corr.plot(x2, y2, label = "auto corr ch 2")
		self.corr.plot(x3, y3, label = "cross-corr")

		self.corr.legend(loc='upper right')

		self.canvas1.draw()

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
					dataset = fcs_importer.Fill_datasets_fcs(lines, fcs_importer.Find_repetitions (lines))

				if filename.endswith('.SIN'): 
					dataset = fcs_importer.Fill_datasets_sin(lines)

				dataset1 = copy.deepcopy(dataset)


				treetree = Data_tree (self.tree, name, dataset.repetitions)
				tree_list.append(treetree)

				tree_list_name.append(name)

				binning_list.append(1)


				data_list_raw.append(dataset)


				data_list_current.append(dataset1)



				repetitions_list.append(dataset.repetitions)

				peaks_list.append([None] * dataset.repetitions)

				root.update()   

		self.pb.destroy()
		self.value_label.destroy()


	def Plot_data(self, event):

		

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

		self.Plot_this_data(data_list_raw[file_index], rep)

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
		self.canvas1.draw()
	
		self.figure1.tight_layout()
	

		data_list_raw = []
		data_list_current = []
		tree_list = []
		tree_list_name = []


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
		self.frame02.pack(side="top", fill="x")



		self.scrollbar = tk.Scrollbar(self.frame02)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame02, width = 100, height = 10)
		self.Datalist.pack(side = "left", anchor = "nw")
		
		
		
		self.tree=CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Plot_data)

		self.Datalist.config(width = 100, height = 10)
		"""self.frame021 = tk.Frame(self.frame02)
		self.frame021.pack(side="top", fill="x")

		self.Norm_button = tk.Button(self.frame021, text="Normalize", command=self.Normalize)
		self.Norm_button.grid(row = 0, column = 0)

		self.Raw_button = tk.Button(self.frame021, text="Restore raw", command=self.Restore)
		self.Raw_button.grid(row = 0, column = 1)

		self.Normalization = ttk.Combobox(self.frame021,values = ["z-score", "mean"] )
		self.Normalization.config(state = "readonly")
		#Threshold.config(font=helv36)
		self.Normalization.grid(row = 0, column = 2)

		self.Normalization.set("z-score")

		self.Normalization.bind("<<ComboboxSelected>>", self.Normalize_index)
		"""

		self.frame023 = tk.Frame(self.frame02)
		self.frame023.pack(side="top", fill="x")


		self.Restruct_button = tk.Button(self.frame023, text="Restructure data", command=Restruct_fun)
		self.Restruct_button.grid(row = 0, column = 0, sticky="W")

		self.Threshold_button = tk.Button(self.frame023, text="Peak analysis", command=Threshold_fun)
		self.Threshold_button.grid(row = 1, column = 0, sticky="W")

		self.Diffusion_button = tk.Button(self.frame023, text="Diffusion analysis", command=Diffusion_fun)
		self.Diffusion_button.grid(row = 2, column = 0, sticky="W")

		self.Add_to_plot_button = tk.Button(self.frame023, text="Plot", command=Which_tab)
		self.Add_to_plot_button.grid(row = 3, column = 0, sticky="W")

		self.Output_button = tk.Button(self.frame023, text="Output", command=Which_tab)
		self.Output_button.grid(row = 4, column = 0, sticky="W")




		self.frame04 = tk.Frame(frame0)
		self.frame04.pack(side="top", fill="x")



		fig_height = (win_height/1.89 - self.Datalist.winfo_height() - self.Import_Button.winfo_height())/dpi_all
		self.figure1 = Figure(figsize=(win_width/(2*dpi_all),fig_height), dpi=100)




		gs = self.figure1.add_gridspec(2, 1)


		self.traces = self.figure1.add_subplot(gs[:1, 0])

		self.traces.set_title("Intensity traces")

		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('intensity (a.u.)')
		self.traces.set_xlabel('Time (s)')

		self.corr = self.figure1.add_subplot(gs[1, 0])



		self.corr.set_title("Correlation curves")

		self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.corr.set_ylabel('G (tau)')
		self.corr.set_xlabel('Delay time')




		self.canvas1 = FigureCanvasTkAgg(self.figure1, self.frame04)
		self.canvas1.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas1, self.frame04)
		self.toolbar.update()
		self.canvas1.get_tk_widget().pack()

		self.figure1.tight_layout()

		self.framepb = tk.Frame(frame0)
		self.framepb.pack(side="top", fill="x")




		

	

class FFP_frame :

	def Put_cross(self):
		main_xlim = ffp.main.get_xlim()
		main_ylim = ffp.main.get_ylim()

		ver = float(self.X_cross.get())
		hor = float(self.Y_cross.get())

		Plot_main()

		self.main.hlines(hor, main_xlim[0], main_xlim[1], color = 'black', zorder=3)
		self.main.vlines(ver,main_ylim[0], main_ylim[1], color = 'black', zorder=3)

		ffp.main.set_xlim(main_xlim)
		ffp.main.set_ylim(main_ylim)

		self.canvas3.draw()

		self.figure3.tight_layout()


	def Type_plot_index(self, event):
		self.plot_type = self.Type_plot.get()

	def __init__(self, frame1, win_width, win_height, dpi_all):

		


		self.frame3 = tk.Frame(frame1)
		self.frame3.pack(side="top", fill="x")

		"""self.X_label = tk.Label(self.frame3, text="X cross: ")
		self.X_label.grid(row = 0, column = 0)

		self.X_cross = tk.Entry(self.frame3, width = 9)
		self.X_cross.grid(row = 0, column = 1)

		

		self.Y_label = tk.Label(self.frame3, text="Y cross: ")
		self.Y_label.grid(row = 1, column = 0)

		self.Y_cross = tk.Entry(self.frame3, width = 9)
		self.Y_cross.grid(row = 1, column = 1)

		self.Put_cross_button = tk.Button(self.frame3, text="Put cross", command=self.Put_cross)
		self.Put_cross_button.grid(row = 0, column = 2, rowspan = 2)"""

		

		self.plot_type = "dot plot"
		self.Type_plot = ttk.Combobox(self.frame3,values = ["dot plot", "density plot"] )
		self.Type_plot.config(state = "readonly")
		#Threshold.config(font=helv36)
		self.Type_plot.grid(row = 0, column = 2)

		self.Type_plot.set("dot plot")

		self.Type_plot.bind("<<ComboboxSelected>>", self.Type_plot_index)
		


		self.frame12 = tk.Frame(frame1)
		self.frame12.pack(side="top", fill="x")


		self.figure3 = Figure(figsize=(win_width/(2*dpi_all),win_width/(2.25*dpi_all)), dpi=100)
		self.main = self.figure3.add_subplot(1, 1, 1)

		self.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.main.set_ylabel('Intensity')
		self.main.set_xlabel('Intensity')


		self.canvas3 = FigureCanvasTkAgg(self.figure3, self.frame12)
		self.canvas3.get_tk_widget().pack()


		self.toolbar = NavigationToolbar2Tk(self.canvas3, self.frame12)
		self.toolbar.update()
		self.canvas3.get_tk_widget().pack()

		self.figure3.tight_layout()

		self.frame13 = tk.Frame(frame1)
		self.frame13.pack(side="top", fill="x")

		self.ffp_btn = tk.Button(self.frame13, text="Configure", command=Norm)
		self.ffp_btn.grid(column = 0, row =0)

		self.ffp_export_btn = tk.Button(self.frame13, text="Export", command=Norm)
		self.ffp_export_btn.grid(column = 0, row =1)


	
class Diff_frame :

	def __init__ (self, frame1, win_width, win_height, dpi_all):

		self.frame12 = tk.Frame(frame1)
		self.frame12.pack(side="top", fill="x")


		self.figure3 = Figure(figsize=(win_width/(2*dpi_all),win_width/(2.25*dpi_all)), dpi=100)
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

		self.ffp_export_btn = tk.Button(self.frame13, text="Export", command=Norm)
		self.ffp_export_btn.grid(column = 0, row =0, sticky = "w")

		self.ffp_btn = tk.Button(self.frame13, text="Configure", command=Norm)
		self.ffp_btn.grid(column = 0, row =1, sticky = "w")
		
class GP_frame :

	def Export_plot(self):
		global fit_list_x
		global fit_list_y


		#cwd = os.getcwd()
		#filepath = cwd + os.path.sep + "Out.txt"
		filepath = "C:\\Users\\taras.sych\\Desktop\\Output\\" + output_file_name + ".txt"
		#filepath1 = "C:\\Users\\taras.sych\\Desktop\\Output\\Out_fit.txt"
		file_export = open(filepath, "w+")

		file_export.write("GP List:" +  "\n" )
		
		for i in range (0, len(gp.gp_all_points)):
			file_export.write(str(gp.gp_all_points[i]) +  "\n" )


		file_export.write("\n" + "GP Histogram:" +  "\n" )

		
   

		for i in range (0, len(gp.gp_histogram)):
			file_export.write(str(gp.gp_xbins[i]) + "\t" +  str(gp.gp_histogram[i]) + "\n" )

	
		#file_export.close()

		
		#file_export = open(filepath1, "w+")

		file_export.write("\n" + "GP Histogram Fit:" +  "\n" )

   

		for i in range (0, len(fit_list_y)):
			file_export.write(str(fit_list_x[i]) + "\t" +  str(fit_list_y[i]) + "\n" )

	
		

		file_export.write("\n" + "Fitting parameters:" +  "\n" )
		
		if (len(Fit_params) == 3):
			file_export.write ( "A: " + str(round (Fit_params[0],3)) + "\n")
			file_export.write ( "Mean: " + str(round(Fit_params[1],3)) + "\n")
			file_export.write ( "Sigma: " + str(round(Fit_params[2],3)) + "\n")

		if (len(Fit_params) == 6):
			file_export.write ( "Peak 1" + "\n")
			file_export.write ( "A: " + str(round (Fit_params[0],3)) + "\n")
			file_export.write ( "Mean: " + str(round(Fit_params[1],3)) + "\n")
			file_export.write ( "Sigma: " + str(round(Fit_params[2],3)) + "\n")

			file_export.write ( "Peak 2" + "\n")
			file_export.write ( "A: " + str(round (Fit_params[3],3)) + "\n")
			file_export.write ( "Mean: " + str(round(Fit_params[4],3)) + "\n")
			file_export.write ( "Sigma: " + str(round(Fit_params[5],3)) + "\n")

		file_export.close()


	"""def Fit_gaus(self):
			
					
					try:
					
						global fit_list_x
						global fit_list_y
						global Fit_params
			
						x = self.gp_xbins
						y = self.gp_histogram
			
						x1 = np.linspace(min(x), max(x), num=500)
			
						if (self.gaus_number == 1):
			
							#m1 = float(self.gaus_mean_1.get())
			
							#popt,pcov = curve_fit(Gauss, x, y, bounds=((-np.inf,-0.8 ,-np.inf), (np.inf, 0.6 ,np.inf)))
							popt,pcov = curve_fit(Gauss, x, y)
			
							#print(popt)
			
			
							Plot_gp()
							fit_list_x = x1
							fit_list_y = Gauss(x1, *popt)
							Fit_params = popt
			
							gp.main.plot(x1, Gauss(x1, *popt), 'r-', label='fit')
			
							gp.canvas3.draw()
			
							gp.figure3.tight_layout()
			
							self.fit_parampampams.delete(0,'end')
			
							self.fit_parampampams.insert(0, "Fitting parameters:") 
							self.fit_parampampams.insert(1, "A:\t" + str(round (popt[0],3)))
							self.fit_parampampams.insert(1, "Mean:\t" + str(round(popt[1],3)))
							self.fit_parampampams.insert(1, "Sigma:\t" + str(round(popt[2],3)))
							
			
						if (self.gaus_number == 2):
			
							popt,pcov = curve_fit(Gauss2, x, y)
			
							Fit_params = popt
			
							popt1 = popt[0:3]
			
							popt2 = popt[3:6]
			
							#print(popt1)
							#print(popt2)
			
							Plot_gp()
							fit_list_x = x1
							fit_list_y = Gauss2(x1, *popt)
			
							gp.main.plot(x1, Gauss(x1, *popt1), color = 'orange', label='fit')
			
							gp.main.plot(x1, Gauss(x1, *popt2), color = 'orange', label='fit')
			
							gp.main.plot(x1, Gauss2(x1, *popt), 'r-', label='fit')
			
							gp.canvas3.draw()
			
							gp.figure3.tight_layout()
			
							self.fit_parampampams.delete(0,'end')
			
							self.fit_parampampams.insert('end', "Fitting parameters:")
							self.fit_parampampams.insert('end', "Peak 1:")
							self.fit_parampampams.insert('end', "A:\t" + str(round (popt[0],3)))
							self.fit_parampampams.insert('end', "Mean:\t" + str(round(popt[1],3)))
							self.fit_parampampams.insert('end', "Sigma:\t" + str(round(popt[2],3)))
							self.fit_parampampams.insert('end', "Peak 2:")
							self.fit_parampampams.insert('end', "A:\t" + str(round (popt[3],3)))
							self.fit_parampampams.insert('end', "Mean:\t" + str(round(popt[4],3)))
							self.fit_parampampams.insert('end', "Sigma:\t" + str(round(popt[5],3)))
			
					except:
						tk.messagebox.showerror(title='Error', message=Message_generator())
			
			
			
			
			
					if (self.gaus_number == 3):
			
						popt,pcov = curve_fit(Gauss3, x, y)
			
						popt1 = popt[0:3]
			
						popt2 = popt[3:6]
			
						popt3 = popt[6:9]
			
						#print(popt1)
						#print(popt2)
						#print(popt3)
			
			
						Plot_gp()
			
						gp.main.plot(x1, Gauss(x1, *popt1), color = 'orange', label='fit')
			
						gp.main.plot(x1, Gauss(x1, *popt2), color = 'orange', label='fit')
			
						gp.main.plot(x1, Gauss(x1, *popt3), color = 'orange', label='fit')
			
						gp.main.plot(x1, Gauss(x1, *popt2), color = 'orange', label='fit')
			
						gp.main.plot(x1, Gauss3(x1, *popt), 'r-', label='fit')
			
						gp.canvas3.draw()
			
						gp.figure3.tight_layout()
			
						self.fit_parampampams.delete(0,'end')"""



	"""def How_many_gaus(self, event):
			
					self.gaus_number = int(self.Gauss_Fit.get())"""

	def __init__(self, frame1, win_width, win_height, dpi_all):

		

		
		



		self.frame12 = tk.Frame(frame1)
		self.frame12.pack(side="top", fill="x")


		self.figure3 = Figure(figsize=(win_width/(2*dpi_all),win_width/(2.25*dpi_all)), dpi=100)
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

		self.ffp_export_btn = tk.Button(self.frame13, text="Export", command=self.Export_plot)
		self.ffp_export_btn.grid(column = 0, row =0, sticky = "w")

		self.ffp_btn = tk.Button(self.frame13, text="Configure", command=Norm)
		self.ffp_btn.grid(column = 0, row =1, sticky = "w")

class GP_Diff_frame:


	def __init__ (self, frame1, win_width, win_height, dpi_all):

		self.frame12 = tk.Frame(frame1)
		self.frame12.pack(side="top", fill="x")


		self.figure3 = Figure(figsize=(win_width/(2*dpi_all),win_width/(2.25*dpi_all)), dpi=100)
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

		self.ffp_export_btn = tk.Button(self.frame13, text="Export", command=Norm)
		self.ffp_export_btn.grid(column = 0, row =0, sticky = "w")

		self.ffp_btn = tk.Button(self.frame13, text="Configure", command=Norm)
		self.ffp_btn.grid(column = 0, row =1, sticky = "w")


class Diffusion_window :

	def Apply_to_all(self):
	
		global rep_index

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()



		for rep_index_i in range (data_list_current[file_index].repetitions): 
			rep_index = rep_index_i
			for param in self.list_of_params:
				self.full_dict[param]["Init"].delete(0,"end")
				self.full_dict[param]["Init"].insert(0,str(round(self.list_of_inits_for_fit_all[param],3)))


			self.Plot_curve()
			self.Fit_corr_curve()

		self.fit_all_flag = False


	def Fit_corr_curve(self):

		
		


		x = self.x_ch1
		y = self.y_ch1

		params = lmfit.Parameters()

		row_index = 1
		for param in self.list_of_params:

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



		data_list_raw[file_index].diff_fitting[rep_index] = output_dict

		#print(data_list_raw[file_index].diff_fitting)
			





		if self.fit_all_flag == False:	

			self.curves.cla()
			self.residuals.cla()
										
										
			self.curves.set_title("Correlation curves")
			self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.curves.set_ylabel('G(tau)')
			self.curves.set_xlabel('Delay time')
			self.curves.set_xscale ('log')
			self.curves.scatter(x, y, label = 'raw')
			
			x1 = np.linspace(min(x), max(x), num=50000)


			if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component':
				self.curves.plot(x, Corr_curve(x, *popt), 'r-', label='fit')
				self.residuals.plot(x, o1.residual, 'b-')
				#self.curves.scatter(x, Corr_curve(x, *popt))



			self.canvas5.draw()

			self.figure5.tight_layout()


	def resid (self, params, x, ydata ):

		param_list = []

		for param in self.list_of_params:

			param_list.append( np.float64(params[param].value))
		

		
		
		if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component':
			y_model = Corr_curve(x, *param_list)


		return y_model - ydata

	def Temp(self, event):
		print(1)

	def Update_plot(self, event):
		self.Plot_curve()


	def Plot_curve(self):


		global file_index
		global rep_index

		if self.fit_all_flag == False:
			self.curves.cla()


		if self.ch_01_var.get() == 1:


			x1 = data_list_raw[file_index].datasets_list[rep_index].channels_list[0].auto_corr_arr.x
			y1 = data_list_raw[file_index].datasets_list[rep_index].channels_list[0].auto_corr_arr.y

			self.x_ch1 = x1
			self.y_ch1 = y1

			if self.fit_all_flag == False:
				self.curves.scatter(x1, y1, label = "auto corr ch 1")


		
		if self.ch_02_var.get() == 1:

			x2 = data_list_raw[file_index].datasets_list[rep_index].channels_list[1].auto_corr_arr.x
			y2 = data_list_raw[file_index].datasets_list[rep_index].channels_list[1].auto_corr_arr.y

			self.x_ch2 = x1
			self.y_ch2 = y1

			if self.fit_all_flag == False:
				self.curves.scatter(x2, y2, label = "auto corr ch 2")


		if self.ch_12_var.get() == 1:

			self.x_ch12 = x1
			self.y_ch12 = y1

			x3 = data_list_raw[file_index].datasets_list[rep_index].cross_list[0].cross_corr_arr.x
			y3 = data_list_raw[file_index].datasets_list[rep_index].cross_list[0].cross_corr_arr.y

			if self.fit_all_flag == False:
				self.curves.scatter(x3, y3, label = "cross-corr")

		
		if self.fit_all_flag == False:
			self.curves.set_title("Correlation curves")
			self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.curves.set_ylabel('G(tau)')
			self.curves.set_xlabel('Delay time')
			self.curves.set_xscale ('log')

			

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



		rep = rep1-1

		self.Plot_curve()

	def Fitting_frame(self):

		self.frame004.destroy()

		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component':

			self.list_of_params = ['offset', 'GN0', 'A', 'txy', 'alpha', 'AR', 'B', 'T_tri' ]
			self.list_of_inits = ['1', '1', '1', '0.02', '1', '5', '1', '0.005']
			self.list_of_min = ['0', '0', '0', '0', '0', '0', '0', '0']
			self.list_of_max = ['10', '5', '1', '100000', '20', '20', '1', '100']

			

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

			row_index+=1








	def __init__(self, win_width, win_height, dpi_all):


		
		self.fit_all_flag = False

		global file_index
		global rep_index

		self.win_diff = tk.Toplevel()

		self.th_width = round(0.7*self.win_diff.winfo_screenwidth())
		self.th_height = round(0.4*self.win_diff.winfo_screenwidth())

		self.line1 = str(self.th_width) + "x" + str(self.th_height)


		self.win_diff.geometry(self.line1)

		self.frame002 = tk.Frame(self.win_diff)
		self.frame002.pack(side = "left", anchor = "nw")

		



		self.scrollbar = tk.Scrollbar(self.frame002)
		self.scrollbar.pack(side = "left", fill = "y")


		self.Datalist = tk.Listbox(self.frame002, width = 100, height = 10)
		self.Datalist.pack(side = "top", anchor = "nw")
		
		
		
		self.tree = CheckboxTreeview(self.Datalist)
		self.tree.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree.pack()


		self.tree.config(yscrollcommand = self.scrollbar.set)
		self.scrollbar.config(command = self.tree.yview)

		self.tree.bind('<<TreeviewSelect>>', self.Choose_curve)



		self.Datalist.config(width = 100, height = 10)


		self.frame003 = tk.Frame(self.frame002)
		self.frame003.pack(side = "top", anchor = "nw")

		self.ch_01_var = tk.IntVar(value=1)
		self.ch_02_var = tk.IntVar(value=1)
		self.ch_12_var = tk.IntVar(value=1)
		self.ch_21_var = tk.IntVar(value=1)

		self.CH_01=tk.Checkbutton(self.frame003, text="CH_01", variable=self.ch_01_var, command=self.Plot_curve)
		self.CH_01.grid(row = 0, column = 0, sticky='w')

		self.CH_02=tk.Checkbutton(self.frame003, text="CH_02", variable=self.ch_02_var, command=self.Plot_curve)
		self.CH_02.grid(row = 0, column = 1, sticky='w')

		self.CH_12=tk.Checkbutton(self.frame003, text="CH_12", variable=self.ch_12_var, command=self.Plot_curve)
		self.CH_12.grid(row = 0, column = 2, sticky='w')

		self.CH_21=tk.Checkbutton(self.frame003, text="CH_21", variable=self.ch_21_var, command=self.Plot_curve)
		self.CH_21.grid(row = 0, column = 3, sticky='w')


		self.frame000 = tk.Frame(self.win_diff)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi=100)
						
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



		




		
		self.frame001 = tk.Frame(self.frame002)
		self.frame001.pack(side = "top", anchor = "nw")

		

		self.Norm_label = tk.Label(self.frame001, text="FCS curve fitting: ")
		self.Norm_label.grid(row = 0, column = 0, columnspan = 2, sticky = 'w')

		self.Triplet = ttk.Combobox(self.frame001,values = ["triplet", "no triplet"], width = 9 )
		self.Triplet.config(state = "readonly")
		
		self.Triplet.grid(row = 1, column = 0, sticky='w')

		self.Triplet.set("triplet")

		self.Triplet.bind("<<ComboboxSelected>>", self.Temp)

		self.Components = ttk.Combobox(self.frame001,values = ["1 component", "2 components", "3 components"], width = 9 )
		self.Components.config(state = "readonly")
		
		self.Components.grid(row = 1, column = 1, sticky='w')

		self.Components.set("1 component")

		self.Components.bind("<<ComboboxSelected>>", self.Temp)



		self.Fit_button = tk.Button(self.frame001, text="Fit", command=self.Fit_corr_curve)
		self.Fit_button.grid(row = 2, column = 0, sticky='ew')



		self.Fit_all_button = tk.Button(self.frame001, text="Fit all", command=self.Apply_to_all)
		self.Fit_all_button.grid(row = 2, column = 1, sticky='ew')

		self.Table_label = tk.Label(self.frame001, text="Fitting parampampams: ")
		self.Table_label.grid(row = 3, column = 0, columnspan = 2, sticky = 'w')



		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		self.Fitting_frame()
		

		global tree_list
		global tree_list_name
		global repetitions_list
		
		

		for i in range(0, len(tree_list_name)):
			name = tree_list_name[i]
			Data_tree (self.tree, name, data_list_current[i].repetitions)


		#self.Plot_curve()



class Threshold_window:

	def Apply_to_all(self):

		global rep_index

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()

		for rep_index_i in range (data_list_current[file_index].repetitions):
			for param in self.list_of_params:
				self.full_dict[param]["Init"].delete(0,"end")
				self.full_dict[param]["Init"].insert(0,str(round(self.list_of_inits_for_fit_all[param],3)))
				
			rep_index = rep_index_i
			self.Peaks()
			self.Fit_gaus()

		self.fit_all_flag = False


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
			self.gp_hist.set_ylabel('Counts')
			self.gp_hist.set_xlabel('GP')
			self.gp_hist.bar(x, y, width = x[1] - x[0], bottom=None, align='center', label = 'raw')
			

			if self.Components.get() == '1 component':
				self.gp_hist.plot(x1, Gauss(x1, *popt), 'r-', label='fit')

			if self.Components.get() == '2 components':
				self.gp_hist.plot(x1, Gauss2(x1, *popt), 'r-', label='fit')
				popt1 = popt[:3]
				popt2 = popt[3:6]
				self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')

			if self.Components.get() == '3 components':
				self.gp_hist.plot(x1, Gauss3(x1, *popt), 'r-', label='fit')
				popt1 = popt[:3]
				popt2 = popt[3:6]
				popt3 = popt[6:9]
				self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, Gauss(x1, *popt3), color = 'yellow', label='fit')



			self.canvas5.draw()

			self.figure5.tight_layout()

	
	def Update_thresholds (self):
		global change_normal
		change_normal = False
		if data_list_raw[file_index].gp_fitting[rep_index] != None:
			data_list_raw[file_index].gp_fitting[rep_index] = None

		self.Peaks()




	def Peaks (self):

		
		


		
		main_xlim = self.peaks.get_xlim()
		main_ylim = self.peaks.get_ylim()

		th1 = data_list_current[file_index].threshold_ch1
		th2 = data_list_current[file_index].threshold_ch2


		int_div = int(rep_index/data_list_current[file_index].binning)

		


		x1 = []
		x2 = []
		y1 = []
		y2 = []
		y1_raw = []
		y2_raw = []


		



		for rep_index_i in range (data_list_current[file_index].repetitions):
						
			if int(rep_index_i/data_list_current[file_index].binning) == int_div:

				#print ("adding repetition ", rep_index_i)


				if len(x1) == 0:
					x_min = 0
				else:
					x_min = max(x1) + x1[1] - x1[0]

				x_temp_1 = [elem + x_min for elem in data_list_current[file_index].datasets_list[rep_index_i].channels_list[0].fluct_arr.x]
				x_temp_2 = [elem + x_min for elem in data_list_current[file_index].datasets_list[rep_index_i].channels_list[1].fluct_arr.x]


				x1.extend(x_temp_1)
				y1.extend(data_list_current[file_index].datasets_list[rep_index_i].channels_list[0].fluct_arr.y)
				y1_raw.extend(data_list_raw[file_index].datasets_list[rep_index_i].channels_list[0].fluct_arr.y)

				x2.extend(x_temp_2)
				y2.extend(data_list_current[file_index].datasets_list[rep_index_i].channels_list[1].fluct_arr.y)
				y2_raw.extend(data_list_raw[file_index].datasets_list[rep_index_i].channels_list[1].fluct_arr.y)




		"""if th1 == 0:
									self.ch1_th.delete(0,"end")
									self.ch1_th.insert(0,str(round(np.mean(y1),2)))
									data_list_current[file_index].threshold_ch1 = round(np.mean(y1),2)
						
								if th2 == 0:
									self.ch2_th.delete(0,"end")
									self.ch2_th.insert(0,str(round(np.mean(y2),2)))
									data_list_current[file_index].threshold_ch2 = round(np.mean(y2),2)"""



		data_list_current[file_index].threshold_ch1 = float(self.ch1_th.get())

		data_list_current[file_index].threshold_ch2 = float(self.ch2_th.get())


		th1 = data_list_current[file_index].threshold_ch1
		th2 = data_list_current[file_index].threshold_ch2

		print ("Thresholds: ", th1, th2)

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

			self.peaks.set_title("Intensity traces")
			
			self.peaks.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.peaks.set_ylabel('Intensity (a.u.)')
			self.peaks.set_xlabel('Time (s)')

			self.hist1.set_title("Intensity histograms")

			self.hist1.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.hist1.set_ylabel('Counts')
			self.hist1.set_xlabel('Intensity (a.u.)')



			if which_channel == "channel 1" or which_channel == "both or" or which_channel == "both and":
				self.peaks.plot(x1, y1, '#1f77b4', zorder=1)
				self.peaks.hlines(th1, min(x1), max(x1), color = 'magenta', zorder=2)
				
				if (self.var.get() == 1):
					self.peaks.plot(xp1, yp1, "x", color = 'magenta', zorder = 3)

				bins_1 = int(np.sqrt(len(yh1)))
				if bins_1 == 0:
					bins_1 = 1
				self.hist1.hist(yh1, bins = bins_1)
				

			if which_channel == "channel 2" or which_channel == "both or" or which_channel == "both and":
				
				self.peaks.plot(x2, y2, '#ff7f0e', zorder=1)
				self.peaks.hlines(th2, min(x1), max(x1), color = 'green', zorder=2)

				if (self.var.get() == 1):
					self.peaks.plot(xp2, yp2, "x", color = 'green', zorder = 3)

				bins_2 = int(np.sqrt(len(yh2)))
				if bins_2 == 0:
					bins_2 = 1
				self.hist1.hist(yh2, bins = bins_2)

			"""if change_normal == False:
													self.peaks.set_xlim(main_xlim)
													self.peaks.set_ylim(main_ylim)"""

		

		gp_list_temp = []

		

		
		for k in range (len(yp1_raw)):
			gp_1 = (yp1_raw[k] - yp2_raw[k])/(yp2_raw[k] + yp1_raw[k])



			if abs(gp_1) < 1:
				gp_list_temp.append(gp_1)



		
		self.n, bins, patches = self.gp_hist.hist(gp_list_temp, bins = int(np.sqrt(len(gp_list_temp))))

			
		

		self.x_bins=[]
		for ii in range (len(bins)-1):
			self.x_bins.append( (bins[ii+1] - bins[ii])/2 + bins[ii])


		if self.fit_all_flag == False:
			self.gp_hist.set_title("GP histogram")
			self.gp_hist.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.gp_hist.set_ylabel('Counts')
			self.gp_hist.set_xlabel('GP')


			if data_list_raw[file_index].gp_fitting[rep_index] != None:


				x1 = np.linspace(min(self.x_bins), max(self.x_bins), num=500)
				popt = []

				for param in data_list_raw[file_index].gp_fitting[rep_index].keys():
			

					popt.append(np.float64(data_list_raw[file_index].gp_fitting[rep_index][param]))

					



				if self.Components.get() == '1 component':
					print("1 comp")
					self.gp_hist.plot(x1, Gauss(x1, *popt), 'r-', label='fit')

				if self.Components.get() == '2 components':
					print("2 comp")
					self.gp_hist.plot(x1, Gauss2(x1, *popt), 'r-', label='fit')
					popt1 = popt[:3]
					popt2 = popt[3:6]
					
					self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
					self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')

				if self.Components.get() == '3 components':
					self.gp_hist.plot(x1, Gauss3(x1, *popt), 'r-', label='fit')
					print("3 comp")
					popt1 = popt[:3]
					popt2 = popt[3:6]
					popt3 = popt[6:9]
					
					self.gp_hist.plot(x1, Gauss(x1, *popt1), color = 'yellow', label='fit')
					self.gp_hist.plot(x1, Gauss(x1, *popt2), color = 'yellow', label='fit')
					self.gp_hist.plot(x1, Gauss(x1, *popt3), color = 'yellow', label='fit')





			self.canvas5.draw()

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
			self.ch1_th.insert(0,str(3))

			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(3))

		if self.normalization_index == "manual":

			x1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.x
			y1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.y

			x2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.x
			y2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.y


			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(round(np.mean(y1),2)))

			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(round(np.mean(y2),2)))

		self.Update_thresholds_button.invoke()


	def Normalize(self):
		

		global file_index
		global rep_index

		


		data_list_current[file_index] = copy.deepcopy(data_list_raw[file_index])


			
		for rep in range (repetitions_list[file_index]):



			y1 = data_list_raw[file_index].datasets_list[rep].channels_list[0].fluct_arr.y
			y2 = data_list_raw[file_index].datasets_list[rep].channels_list[1].fluct_arr.y
			
				
			if self.normalization_index == "z-score":
				y1z = stats.zscore(y1)
				y2z = stats.zscore(y2)


				data_list_current[file_index].threshold_ch1 = float(self.ch1_th.get())
				data_list_current[file_index].threshold_ch2 = float(self.ch2_th.get())

				data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y = y1z
				data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y = y2z



			


			if self.normalization_index == "manual":



				y1m = y1/np.mean(y1)
				y2m = y2/np.mean(y2)

				

				data_list_current[file_index].threshold_ch1 = float(self.ch1_th.get())
				data_list_current[file_index].threshold_ch2 = float(self.ch2_th.get())
				

				data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y = y1m
				data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y = y2m





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
		
		self.Normalize()



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

		current_repetitions_number = data_list_current[file_index].repetitions

		#print(current_repetitions_number)

		divisors = []
		for divdiv in range(1, current_repetitions_number+1):
			if current_repetitions_number % divdiv == 0:
				divisors.append(divdiv)

		self.Binning_choice.config(values = divisors)
		self.Binning_choice.set(data_list_current[file_index].binning)


		rep = rep1-1


		if data_list_raw[file_index].gp_fitting[rep_index] != None:


			if len(data_list_raw[file_index].gp_fitting[rep_index].keys()) == 3:

				

				self.Components.set("1 component")

			if len(data_list_raw[file_index].gp_fitting[rep_index].keys()) == 6:

				

				self.Components.set("2 components")

			if len(data_list_raw[file_index].gp_fitting[rep_index].keys()) == 9:

				self.Components.set("3 components")

		self.Normalize()



		self.Fitting_frame()



	
	def Binning(self, event):
		global file_index
		global rep_index

		global change_normal

		change_normal = True


		data_list_current[file_index].binning = int(self.Binning_choice.get())

		

		self.Peaks()

	def __init__(self, win_width, win_height, dpi_all):


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


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi=100)
						
		gs = self.figure5.add_gridspec(2, 2)


		self.peaks = self.figure5.add_subplot(gs[0, :2])

		self.peaks.set_title("Intensity traces")

		self.peaks.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.peaks.set_ylabel('Intensity (a.u.)')
		self.peaks.set_xlabel('Time (s)')

		self.hist1 = self.figure5.add_subplot(gs[1, 0])

		self.hist1.set_title("Intensity histogram")

		self.hist1.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.hist1.set_ylabel('Counts')
		self.hist1.set_xlabel('Intensity (a.u.)')


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

		

		self.Binning_label = tk.Label(self.frame001, text="Binning: ")
		self.Binning_label.grid(row = 0, column = 0, sticky = 'w')

		divisors = []


		self.Binning_choice = ttk.Combobox(self.frame001,values = divisors, width = 4 )
		self.Binning_choice.config(state = "readonly")
		
		self.Binning_choice.grid(row = 0, column = 1, sticky = 'w')

		

		self.Binning_choice.bind("<<ComboboxSelected>>", self.Binning)


		self.Norm_label = tk.Label(self.frame001, text="Use for plot: ")
		self.Norm_label.grid(row = 1, column = 0, sticky = 'w')

		self.Normalization_for_plot = ttk.Combobox(self.frame001,values = ["raw", "normalized"], width = 9 )
		self.Normalization_for_plot.config(state = "readonly")
		
		self.Normalization_for_plot.grid(row = 1, column = 1)

		self.Normalization_for_plot.set("raw")

		self.Normalization_for_plot.bind("<<ComboboxSelected>>", self.Normalize_for_plot_index)

		self.var = tk.IntVar()

		self.Peaks_button=tk.Checkbutton(self.frame001, text="Display peaks", variable=self.var, command=self.Update_thresholds)
		self.Peaks_button.grid(row = 1, column = 3, sticky='w')



		




		self.Type_label = tk.Label(self.frame001, text="Detect: ")
		self.Type_label.grid(row = 2, column = 0, sticky='w')

	

		self.Threshold = ttk.Combobox(self.frame001,values = ["both and", "both or", "channel 1", "channel 2"], width = 9 )
		self.Threshold.config(state = "readonly")
		self.Threshold.grid(row = 2, column = 1)

		self.Threshold.set("both and")

		self.Threshold.bind("<<ComboboxSelected>>", self.Threshold_callback)

		
	
		
		self.Norm_label = tk.Label(self.frame001, text="Thresholding: ")
		self.Norm_label.grid(row = 3, column = 0)

		self.Normalization = ttk.Combobox(self.frame001,values = ["manual", "z-score"], width = 9 )
		self.Normalization.config(state = "readonly")
								#Threshold.config(font=helv36)
		self.Normalization.grid(row = 3, column = 1, sticky = 'w')
						
		self.Normalization.set("z-score")
						
		self.Normalization.bind("<<ComboboxSelected>>", self.Normalize_index)


		





		self.ch1_label = tk.Label(self.frame001, text="channel 1: ")
		self.ch1_label.grid(row = 4, column = 0, sticky='w')

		self.ch1_th = tk.Entry(self.frame001, width = 9)
		self.ch1_th.grid(row = 4, column = 1, sticky='w')

		self.ch1_th.insert("end", str(3))

		self.ch2_label = tk.Label(self.frame001, text="channel 2: ")
		self.ch2_label.grid(row = 5, column = 0, sticky='w')

		

		self.ch2_th = tk.Entry(self.frame001, width = 9)
		self.ch2_th.grid(row = 5, column = 1, sticky='w')

		self.ch2_th.insert("end", str(3))


		self.Update_thresholds_button = tk.Button(self.frame001, text="Update thresholds", command=self.Update_thresholds)
		self.Update_thresholds_button.grid(row = 6, column = 0, columnspan = 2, sticky='w')

		self.Put_mean_button = tk.Button(self.frame001, text="Set to default", command=self.Put_default)
		self.Put_mean_button.grid(row = 7, column = 0, columnspan = 2, sticky='w')







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



		self.Fit_button = tk.Button(self.frame007, text="Fit", command=self.Fit_gaus)
		self.Fit_button.grid(row = 0, column = 0, sticky='ew')

		self.Fit_all_button = tk.Button(self.frame007, text="Fit all", command=self.Apply_to_all)
		self.Fit_all_button.grid(row = 0, column = 1, sticky='ew')

		self.Components = ttk.Combobox(self.frame007,values = ["1 component", "2 components", "3 components"], width = 13 )
		self.Components.config(state = "readonly")
		self.Components.grid(row = 0, column = 0, columnspan = 2, sticky='w')
		self.Components.set("1 component")

		self.Components.bind("<<ComboboxSelected>>", self.Choose_components)

		self.Param_label = tk.Label(self.frame007, text="Fitting parampampams:")
		self.Components.grid(row = 1, column = 0, sticky='w', columnspan = 2)


		self.frame004 = tk.Frame(self.frame002)
		self.frame004.pack(side = "top", anchor = "nw")

		self.Fitting_frame()


		for i in range(0, len(tree_list_name)):
			name = tree_list_name[i]
			Data_tree (self.tree_t, name, data_list_current[i].repetitions)

		



	def Temp(self):
		print(1)






	


class Data_tree:

	

	def __init__(self, tree, name, repetitions):
		

		
		

		self.folder1=tree.insert( "", "end", text=name)
		child_id = tree.get_children()[-1]
		for i in range(0, repetitions):
			text1 = "repetition " + str (i+1)
			tree.insert(self.folder1, "end", text=text1)

		tree.focus(child_id)
		tree.selection_set(child_id)


			
		
	
			


gp_list = []

peaks_list = []

data_list_raw = []


data_list_current = []

repetitions_list = []


root = tk.Tk()
root.title("PFF analysis")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

win_width = round(0.9 * screen_width)
win_height = round (0.9 * screen_height)

fontsize = round(win_width/85)

helv36 = tkFont.Font(size=fontsize)

line = str(win_width) + "x" + str(win_height)


root.geometry(line)

dpi_all = 100


frame0 = tk.LabelFrame(root)
frame0.pack(side = "left", anchor = "nw", expand =1, fill=tk.BOTH)
frame0.config(bd=0, width = round(win_width/2), height = win_height)
frame0.grid_propagate(0)

tabs = ttk.Notebook(root, width=round(win_width/2), height=win_height, padding = 0)

tab = []


frame1=ttk.Frame(tabs)
frame4 = ttk.Frame(tabs)
frame5 = ttk.Frame(tabs)
frame6 = ttk.Frame(tabs)

tabs.add(frame1, text = "Diffusion plot")
tabs.add(frame4, text = "GP plot")
tabs.add(frame5, text = "Diffusion vs GP")
tabs.add(frame6, text = "Dot plot")
tabs_number = 4;

tabs.pack(side = "left", anchor = "nw")



data_frame = Left_frame(frame0, win_width, win_height, dpi_all )

#ffp = FFP_frame(frame1, win_width, win_height, dpi_all)

diff = Diff_frame(frame1, win_width, win_height, dpi_all)
gp = GP_frame(frame4, win_width, win_height, dpi_all)
gp_diff = GP_Diff_frame(frame5, win_width, win_height, dpi_all)

root.mainloop()