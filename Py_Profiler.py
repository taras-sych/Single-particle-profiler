import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2TkAgg)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib import cm as mplcm

from ttkwidgets import CheckboxTreeview

import fcs_importer

import codecs

import os

from scipy import stats

import copy

import numpy as np

from scipy.signal import find_peaks

from scipy.optimize import curve_fit
import random


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

		
		for k in range (len(peaks_list[file1][rep1].x)):
			gp_1 = (peaks_list[file1][rep1].x[k] - peaks_list[file1][rep1].y[k])/(peaks_list[file1][rep1].x[k] + peaks_list[file1][rep1].y[k])



			if abs(gp_1) != 1:
				gp_list.append(gp_1)
				#print (peaks_list[file1][rep1].x[k], peaks_list[file1][rep1].y[k], gp_1)
		



	gp.gp_all_points = copy.deepcopy(gp_list)
	
	n, bins, patches = gp.main.hist(gp_list, bins = 'auto')

	
	x_bins=[]
	for ii in range (len(bins)-1):
		x_bins.append( (bins[ii+1] - bins[ii])/2 + bins[ii])



	
	gp.gp_histogram = copy.deepcopy(n)

	gp.gp_xbins = copy.deepcopy(x_bins)


	

	

		

	gp.main.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
	gp.main.set_ylabel('Counts')
	gp.main.set_xlabel('GP')
	

	

def Which_tab():

	try:
		if tabs.index(tabs.select()) == 0:
			Plot_main()

			ffp.canvas3.draw()

			ffp.figure3.tight_layout()
			

		if tabs.index(tabs.select()) == 1:
			Plot_gp()

			gp.canvas3.draw()

			gp.figure3.tight_layout()

	except:
		tk.messagebox.showerror(title='Error', message=Message_generator())



def Threshold_fun():

	if len(tree_list_name) > 0:

		th_win = Threshold_window(win_width, win_height, dpi_all)

	if len(tree_list_name) == 0:

		tk.messagebox.showerror(title='Error', message=Message_generator())



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
		
		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('Intensity (a.u.)')
		self.traces.set_xlabel('Time (s)')

		self.traces.plot(x1, y1, label = "channel 1")
		self.traces.plot(x2, y2, label = "channel 2")

		self.traces.legend(loc='upper right')

		self.canvas1.draw()

		self.figure1.tight_layout()



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

		self.canvas2.draw()

		self.figure2.tight_layout()




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
		self.canvas2.draw()
		self.figure1.tight_layout()
		self.figure2.tight_layout()

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

		self.Threshold_button = tk.Button(self.frame023, text="Peak analysis", command=Threshold_fun)
		self.Threshold_button.grid(row = 0, column = 0, sticky="W")

		self.Diffusion_button = tk.Button(self.frame023, text="Diffusion analysis", command=Norm)
		self.Diffusion_button.grid(row = 1, column = 0, sticky="W")

		self.Add_to_plot_button = tk.Button(self.frame023, text="Plot", command=Which_tab)
		self.Add_to_plot_button.grid(row = 2, column = 0, sticky="W")

		self.Output_button = tk.Button(self.frame023, text="Output", command=Which_tab)
		self.Output_button.grid(row = 3, column = 0, sticky="W")




		self.frame04 = tk.Frame(frame0)
		self.frame04.pack(side="top", fill="x")


		self.figure1 = Figure(figsize=(win_width/(2*dpi_all),win_height/(4*dpi_all)), dpi=100)
		self.traces = self.figure1.add_subplot(1, 1, 1)

		self.traces.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.traces.set_ylabel('Intensity (a.u.)')
		self.traces.set_xlabel('Time (s)')


		self.canvas1 = FigureCanvasTkAgg(self.figure1, self.frame04)
		self.canvas1.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas1, self.frame04)
		self.toolbar.update()
		self.canvas1.get_tk_widget().pack()

		self.figure1.tight_layout()


		self.frame06 = tk.Frame(frame0)
		self.frame06.pack(side="top", fill="x")


		self.figure2 = Figure(figsize=(win_width/(2*dpi_all),win_height/(4*dpi_all)), dpi=100)
		self.corr = self.figure2.add_subplot(1, 1, 1)

		self.corr.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.corr.set_ylabel('G(tau)')
		self.corr.set_xlabel('Delay time')


		self.canvas2 = FigureCanvasTkAgg(self.figure2, self.frame06)
		self.canvas2.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas2, self.frame06)
		self.toolbar.update()
		self.canvas2.get_tk_widget().pack()

		self.figure2.tight_layout()

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


	def Fit_gaus(self):

		
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

			self.fit_parampampams.delete(0,'end')



	def How_many_gaus(self, event):
		self.gaus_number = int(self.Gauss_Fit.get())

	def __init__(self, frame1, win_width, win_height, dpi_all):

		

		self.gp_histogram = []

		self.gp_xbins = []
		
		self.frame3 = tk.Frame(frame1)
		self.frame3.pack(side="top", fill="x")

		self.Label1 = tk.Label(self.frame3, text="Gauss Fit: ")
		self.Label1.grid(row = 0, column = 0)

		self.Gauss_Fit = ttk.Combobox(self.frame3,values = ["1", "2", "3"], width = 9)
		self.Gauss_Fit.config(state = "readonly")
		self.Gauss_Fit.grid(row = 0, column = 1, sticky = "w")

		self.Gauss_Fit.set("1")

		self.gaus_number = 1

		self.Gauss_Fit.bind("<<ComboboxSelected>>", self.How_many_gaus)

		self.Fit_button = tk.Button(self.frame3, text="Fit", command=self.Fit_gaus, width = 9)
		self.Fit_button.grid(row = 0, column = 2, sticky = "w")

		self.Label2 = tk.Label(self.frame3, text="mean 1: ")
		self.Label2.grid(row = 1, column = 0, sticky = "w")

		self.gaus_mean_1 = tk.Entry(self.frame3, width = 9)
		self.gaus_mean_1.grid(row = 1, column = 1, sticky = "w")

		self.Label3 = tk.Label(self.frame3, text="mean 2: ")
		self.Label3.grid(row = 2, column = 0, sticky = "w")

		self.gaus_mean_2 = tk.Entry(self.frame3, width = 9)
		self.gaus_mean_2.grid(row = 2, column = 1, sticky = "w")

		self.Label4 = tk.Label(self.frame3, text="mean 3: ")
		self.Label4.grid(row = 3, column = 0, sticky = "w")

		self.gaus_mean_3 = tk.Entry(self.frame3, width = 9)
		self.gaus_mean_3.grid(row = 3, column = 1, sticky = "w")

		self.fit_parampampams = tk.Listbox(self.frame3, width = 18, height = 6)
		self.fit_parampampams.grid (row = 1, column = 2, rowspan = 3, sticky = "w")


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


		


class Threshold_window:



	def Initial_plot (self):

		#print(self.var.get())

		th1 = data_list_current[file_index].threshold_ch1
		th2 = data_list_current[file_index].threshold_ch2


		

		x1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.x
		y1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.y

		x2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.x
		y2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.y

		if th1 == 0:
			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(round(np.mean(y1),2)))

		if th2 == 0:
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(round(np.mean(y2),2)))



		data_list_current[file_index].threshold_ch1 = float(self.ch1_th.get())

		data_list_current[file_index].threshold_ch2 = float(self.ch2_th.get())


		th1 = data_list_current[file_index].threshold_ch1
		th2 = data_list_current[file_index].threshold_ch2

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

		xp1 = []
		yp1 = []
		yp2_1 = []
		for p in peaks1:
			xp1.append(x1[p])
			yp1.append(y1[p])
			yp2_1.append(y2[p])


		peaks1, _ = find_peaks(y2, height=th2)

		xp2 = []
		yp2 = []
		yp1_2 = []

		for p in peaks1:
			xp2.append(x2[p])
			yp2.append(y2[p])
			yp1_2.append(y1[p])
		


		self.peaks.cla()
		self.hist1.cla()

		self.peaks.set_title("Intensity traces")
		self.hist1.set_title("Intensity histogram")
		
		self.peaks.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.peaks.set_ylabel('Intensity (a.u.)')
		self.peaks.set_xlabel('Time (s)')

		self.hist1.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.hist1.set_ylabel('Counts')
		self.hist1.set_xlabel('Intensity (a.u.)')

		if which_channel == "both":

			

			self.peaks.plot(x1, y1, zorder=1)
			self.peaks.plot(x2, y2, zorder=2)

			self.peaks.hlines(th1, min(x1), max(x1), color = 'magenta', zorder=3)
			self.peaks.hlines(th2, min(x1), max(x1), color = 'green', zorder=4)

			
			if (self.var.get() == 1):
				self.peaks.plot(xp1, yp1, "x", color = 'magenta', zorder = 5)
				self.peaks.plot(xp2, yp2, "x", color = 'green', zorder = 6)


			

			self.hist1.hist(yh1, bins = 150)
			self.hist1.hist(yh2, bins = 150)

		if which_channel == "channel 1":
			self.peaks.plot(x1, y1, '#1f77b4', zorder=1)
			self.peaks.hlines(th1, min(x1), max(x1), color = 'magenta', zorder=2)
			
			if (self.var.get() == 1):
				self.peaks.plot(xp1, yp1, "x", color = 'magenta', zorder = 3)
			self.hist1.hist(yh1, bins = 150)
			

		if which_channel == "channel 2":
			
			self.peaks.plot(x2, y2, '#ff7f0e', zorder=1)
			self.peaks.hlines(th2, min(x1), max(x1), color = 'green', zorder=2)

			if (self.var.get() == 1):
				self.peaks.plot(xp2, yp2, "x", color = 'green', zorder = 3)
			self.hist1.hist(yh2, bins = 150)

		self.canvas5.draw()

		self.figure5.tight_layout()

		self.Peaks()

		
	
	def Update_thresholds (self):
		global change_normal
		change_normal = False
		self.Peaks()


	def Peaks (self):
		global change_normal

		#print(self.var.get())
		main_xlim = self.peaks.get_xlim()
		main_ylim = self.peaks.get_ylim()

		th1 = data_list_current[file_index].threshold_ch1
		th2 = data_list_current[file_index].threshold_ch2


		

		x1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.x
		y1 = data_list_current[file_index].datasets_list[rep_index].channels_list[0].fluct_arr.y

		x2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.x
		y2 = data_list_current[file_index].datasets_list[rep_index].channels_list[1].fluct_arr.y

		if th1 == 0:
			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(round(np.mean(y1),2)))
			data_list_current[file_index].threshold_ch1 = round(np.mean(y1),2)

		if th2 == 0:
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(round(np.mean(y2),2)))
			data_list_current[file_index].threshold_ch2 = round(np.mean(y2),2)



		data_list_current[file_index].threshold_ch1 = float(self.ch1_th.get())

		data_list_current[file_index].threshold_ch2 = float(self.ch2_th.get())


		th1 = data_list_current[file_index].threshold_ch1
		th2 = data_list_current[file_index].threshold_ch2

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

		xp1 = []
		yp1 = []
		yp2_1 = []
		for p in peaks1:
			xp1.append(x1[p])
			yp1.append(y1[p])
			yp2_1.append(y2[p])


		peaks1, _ = find_peaks(y2, height=th2)

		xp2 = []
		yp2 = []
		yp1_2 = []

		for p in peaks1:
			xp2.append(x2[p])
			yp2.append(y2[p])
			yp1_2.append(y1[p])
		


		self.peaks.cla()
		self.hist1.cla()
		self.gp_hist.cla()
		
		self.peaks.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.peaks.set_ylabel('Intensity (a.u.)')
		self.peaks.set_xlabel('Time (s)')

		self.hist1.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.hist1.set_ylabel('Counts')
		self.hist1.set_xlabel('Intensity (a.u.)')


		if which_channel == "both":

			

			self.peaks.plot(x1, y1, zorder=1)
			self.peaks.plot(x2, y2, zorder=2)

			self.peaks.hlines(th1, min(x1), max(x1), color = 'magenta', zorder=3)
			self.peaks.hlines(th2, min(x1), max(x1), color = 'green', zorder=4)

			
			if (self.var.get() == 1):
				self.peaks.plot(xp1, yp1, "x", color = 'magenta', zorder = 5)
				self.peaks.plot(xp2, yp2, "x", color = 'green', zorder = 6)


			

			self.hist1.hist(yh1, bins = 150)
			self.hist1.hist(yh2, bins = 150)

		if which_channel == "channel 1":
			self.peaks.plot(x1, y1, '#1f77b4', zorder=1)
			self.peaks.hlines(th1, min(x1), max(x1), color = 'magenta', zorder=2)
			
			if (self.var.get() == 1):
				self.peaks.plot(xp1, yp1, "x", color = 'magenta', zorder = 3)
			self.hist1.hist(yh1, bins = 150)
			

		if which_channel == "channel 2":
			
			self.peaks.plot(x2, y2, '#ff7f0e', zorder=1)
			self.peaks.hlines(th2, min(x1), max(x1), color = 'green', zorder=2)

			if (self.var.get() == 1):
				self.peaks.plot(xp2, yp2, "x", color = 'green', zorder = 3)
			self.hist1.hist(yh2, bins = 150)

		if change_normal == False:
			self.peaks.set_xlim(main_xlim)
			self.peaks.set_ylim(main_ylim)

		

		gp_list_temp = []

		for k in range (len(yp1)):
			gp_1 = (yp2_1[k] - yp1[k])/(yp2_1[k] + yp1[k])



			if abs(gp_1) < 1:
				gp_list_temp.append(gp_1)

		for k in range (len(yp2)):
			gp_1 = (yp1_2[k] - yp2[k])/(yp1_2[k] + yp2[k])



			if abs(gp_1) < 1:
				gp_list_temp.append(gp_1)


		self.gp_hist.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.gp_hist.set_ylabel('Counts')
		self.gp_hist.set_xlabel('GP')

		self.gp_hist.hist(gp_list_temp)
		

		self.canvas5.draw()

		self.figure5.tight_layout()


	def Threshold_callback(self, event):

			self.Peaks()


	def Apply(self):
		global file_index

		data_list_current[file_index].threshold_ch1 = float(self.ch1_th.get())

		data_list_current[file_index].threshold_ch2 = float(self.ch2_th.get())

		


		th1 = data_list_current[file_index].threshold_ch1
		th2 = data_list_current[file_index].threshold_ch2

		




		which_channel = self.Threshold.get()


		


		for i in range (data_list_current[file_index].repetitions):
			#print(i)
	


			x1 = data_list_current[file_index].datasets_list[i].channels_list[0].fluct_arr.x
			y1c = data_list_current[file_index].datasets_list[i].channels_list[0].fluct_arr.y
			y1r = data_list_raw[file_index].datasets_list[i].channels_list[0].fluct_arr.y

			x2 = data_list_current[file_index].datasets_list[i].channels_list[1].fluct_arr.x
			y2c = data_list_current[file_index].datasets_list[i].channels_list[1].fluct_arr.y
			y2r = data_list_raw[file_index].datasets_list[i].channels_list[1].fluct_arr.y


			peaks1, _ = find_peaks(y1c, height=th1)

			peaks2, _ = find_peaks(y2c, height=th2)


		
			if self.normalization_index_for_plot == "raw":



				if which_channel == "both":
					

					xp1 = []
					yp1 = []
					yp2_1 = []
					for p in peaks1:
						xp1.append(x1[p])
						yp1.append(y1r[p])
						yp2_1.append(y2r[p])


					xp2 = []
					yp2 = []
					yp1_2 = []

					for p in peaks2:
						xp2.append(x2[p])
						yp2.append(y2r[p])
						yp1_2.append(y1r[p])

					

					X_all = yp1 + yp1_2
					Y_all = yp2_1 + yp2
					T_all = xp1 + xp2

					


				if which_channel == "channel 1":
					

					xp1 = []
					yp1 = []
					yp2_1 = []
					for p in peaks1:
						xp1.append(x1[p])
						yp1.append(y1r[p])
						yp2_1.append(y2r[p])


					X_all = yp1
					Y_all = yp2_1
					T_all = xp1

				if which_channel == "channel 2":


					

					xp2 = []
					yp2 = []
					yp1_2 = []

					for p in peaks2:
						xp2.append(x2[p])
						yp2.append(y2r[p])
						yp1_2.append(y1r[p])

					X_all = yp1_2
					Y_all = yp2
					T_all = xp2






			if self.normalization_index_for_plot == "normalized":

				if which_channel == "both":
					

					xp1 = []
					yp1 = []
					yp2_1 = []
					for p in peaks1:
						xp1.append(x1[p])
						yp1.append(y1c[p])
						yp2_1.append(y2c[p])


					xp2 = []
					yp2 = []
					yp1_2 = []

					for p in peaks2:
						xp2.append(x2[p])
						yp2.append(y2c[p])
						yp1_2.append(y1c[p])

					

					X_all = yp1 + yp1_2
					Y_all = yp2_1 + yp2
					T_all = xp1 + xp2

					


				if which_channel == "channel 1":
					

					xp1 = []
					yp1 = []
					yp2_1 = []
					for p in peaks1:
						xp1.append(x1[p])
						yp1.append(y1c[p])
						yp2_1.append(y2c[p])


					X_all = yp1
					Y_all = yp2_1
					T_all = xp1

				if which_channel == "channel 2":


					

					xp2 = []
					yp2 = []
					yp1_2 = []

					for p in peaks2:
						xp2.append(x2[p])
						yp2.append(y2c[p])
						yp1_2.append(y1c[p])

					X_all = yp1_2
					Y_all = yp2
					T_all = xp2



			#print(X_all)

			peaks_list[file_index][i] = copy.deepcopy(Found_peaks(X_all, Y_all, T_all))

			#print(peaks_list[file_index][i].x)

			#self.win_threshold.destroy()

	def Put_default(self):

		if self.normalization_index == "z-score":
			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(1))

			
			self.ch2_th.delete(0,"end")
			self.ch2_th.insert(0,str(1))

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
		global change_normal
		global file_index
		global rep_index

		change_normal = True

			
		for rep in range (repetitions_list[file_index]):

			y1 = data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y
			y2 = data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y
			
				
			if self.normalization_index == "z-score":
				y1z = stats.zscore(y1)
				y2z = stats.zscore(y2)

				data_list_current[file_index].threshold_ch1 = 1
				data_list_current[file_index].threshold_ch2 = 1

				data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y = y1z
				data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y = y2z

				self.ch1_th.delete(0,"end")
				self.ch1_th.insert(0,str(1))
		
				self.ch2_th.delete(0,"end")
				self.ch2_th.insert(0,str(1))

				self.Peaks()


			if self.normalization_index == "manual":

				y1 = y1/np.mean(y1)
				y2 = y2/np.mean(y2)

				data_list_current[file_index] = copy.deepcopy(data_list_raw[file_index])

				data_list_current[file_index].threshold_ch1 = 0
				data_list_current[file_index].threshold_ch2 = 0

				self.Peaks()

		#self.Update_thresholds_button.invoke()


	def Normalize_index(self, event):

		self.normalization_index = self.Normalization.get()
		
		if self.normalization_index == "z-score":
			#print (self.normalization_index)
			self.Normalize()


		if self.normalization_index == "manual":
			#print (self.normalization_index)
			self.Normalize()

	def Normalize_for_plot_index(self, event):
		self.normalization_index_for_plot = self.Normalization_for_plot.get()
		#print (self.normalization_index_for_plot)


	def Thresholding_type_selection(self, value):
		print(value)

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

		self.Peaks()

	def __init__(self, win_width, win_height, dpi_all):

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

		



		self.scrollbar_t = tk.Scrollbar(self.frame002)
		self.scrollbar_t.pack(side = "left", fill = "y")


		self.Datalist_t = tk.Listbox(self.frame002, width = 100, height = 10)
		self.Datalist_t.pack(side = "top", anchor = "nw")
		
		
		
		self.tree_t = CheckboxTreeview(self.Datalist_t)
		self.tree_t.heading("#0",text="Imported datasets",anchor=tk.W)
		self.tree_t.pack()


		self.tree_t.config(yscrollcommand = self.scrollbar_t.set)
		self.scrollbar_t.config(command = self.tree_t.yview)

		self.tree_t.bind('<<TreeviewSelect>>', self.Plot_trace)

		#self.tree_t.bind('<<>>', )

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



		




		


		

		self.Norm_label = tk.Label(self.frame001, text="Use for plot: ")
		self.Norm_label.grid(row = 0, column = 0, sticky = 'w')

		self.Normalization_for_plot = ttk.Combobox(self.frame001,values = ["raw", "normalized"], width = 9 )
		self.Normalization_for_plot.config(state = "readonly")
		
		self.Normalization_for_plot.grid(row = 0, column = 1)

		self.Normalization_for_plot.set("raw")

		self.Normalization_for_plot.bind("<<ComboboxSelected>>", self.Normalize_for_plot_index)

		self.var = tk.IntVar()

		self.Peaks_button=tk.Checkbutton(self.frame001, text="Display peaks", variable=self.var, command=self.Update_thresholds)
		self.Peaks_button.grid(row = 0, column = 3, sticky='w')

		self.Apply_button = tk.Button(self.frame001, text="Apply and Fit", command=self.Apply)
		self.Apply_button.grid(row = 2, column = 3)

		




		self.Type_label = tk.Label(self.frame001, text="Detect: ")
		self.Type_label.grid(row = 2, column = 0, sticky='w')

	

		self.Threshold = ttk.Combobox(self.frame001,values = ["both", "channel 1", "channel 2"], width = 9 )
		self.Threshold.config(state = "readonly")
		self.Threshold.grid(row = 2, column = 1)

		self.Threshold.set("both")

		self.Threshold.bind("<<ComboboxSelected>>", self.Threshold_callback)

		
	
		
		self.Norm_label = tk.Label(self.frame001, text="Thresholding: ")
		self.Norm_label.grid(row = 3, column = 0)

		self.Normalization = ttk.Combobox(self.frame001,values = ["manual", "z-score"], width = 9 )
		self.Normalization.config(state = "readonly")
								#Threshold.config(font=helv36)
		self.Normalization.grid(row = 3, column = 1, sticky = 'w')
						
		self.Normalization.set("manual")
						
		self.Normalization.bind("<<ComboboxSelected>>", self.Normalize_index)


		"""self.thresholding_type = tk.IntVar()
						
								self.thresholding_type.set(1)
						
								tk.Radiobutton(self.frame001, text = "Manual thresholding", variable = self.thresholding_type, value = 1, command = lambda: self.Thresholding_type_selection(self.thresholding_type.get())).grid(row = 2, column = 0, columnspan = 2, sticky='w')
								tk.Radiobutton(self.frame001, text = "z score", variable = self.thresholding_type, value = 2, command = lambda: self.Thresholding_type_selection(self.thresholding_type.get())).grid(row = 2, column = 3, columnspan = 2, sticky='w')"""






		self.ch1_label = tk.Label(self.frame001, text="channel 1: ")
		self.ch1_label.grid(row = 4, column = 0, sticky='w')

		self.ch1_th = tk.Entry(self.frame001, width = 9)
		self.ch1_th.grid(row = 4, column = 1, sticky='w')

		self.ch1_th.insert("end", str(data_list_current[file_index].threshold_ch1))

		self.ch2_label = tk.Label(self.frame001, text="channel 2: ")
		self.ch2_label.grid(row = 5, column = 0, sticky='w')

		

		self.ch2_th = tk.Entry(self.frame001, width = 9)
		self.ch2_th.grid(row = 5, column = 1, sticky='w')

		self.ch2_th.insert("end", str(data_list_current[file_index].threshold_ch2))


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


		global change_normal
		change_normal = True
		

		self.Peaks()



		global tree_list
		global tree_list_name
		global repetitions_list
		
		

		for i in range(0, len(tree_list_name)):
			name = tree_list_name[i]
			Data_tree (self.tree_t, name, data_list_current[i].repetitions)

		

		"""self.Norm_button = tk.Button(self.frame001, text="Normalize", command=self.Normalize)
								self.Norm_button.grid(row = 3, column = 0)
						
								self.Normalization = ttk.Combobox(self.frame001,values = ["raw", "z-score", "mean" ] )
								self.Normalization.config(state = "readonly")
								#Threshold.config(font=helv36)
								self.Normalization.grid(row = 3, column = 1)
						
								self.Normalization.set("raw")
						
								self.Normalization.bind("<<ComboboxSelected>>", self.Normalize_index)"""










	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#-------------------------------------------------------------------------------------------------------------------------------------------------------
	#-------------------------------------------------------------------------------------------------------------------------------------------------------




class Fitting_window:

	def __init__(self, win_width, win_height, dpi_all):


		self.th_width = round(0.7*self.win_threshold.winfo_screenwidth())
		self.th_height = round(0.4*self.win_threshold.winfo_screenwidth())

		self.line1 = str(self.th_width) + "x" + str(self.th_height)


		self.win_threshold.geometry(self.line1)

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

		self.frame02 = tk.Frame(frame0)
		self.frame02.pack(side="top", fill="x")



	


class Data_tree:

	

	def __init__(self, tree, name, repetitions):
		

		
		

		self.folder1=tree.insert( "", "end", text=name)
		child_id = tree.get_children()[-1]
		for i in range(0, repetitions):
			text1 = "repetition " + str (i+1)
			tree.insert(self.folder1, "end", text=text1)

		tree.focus(child_id)
		tree.selection_set(child_id)



class Found_peaks:
	def __init__(self, x1, y1, t1):
		self.x = x1
		self.y = y1
		self.t = t1
		self.active = True
			
		
	
			


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

tabs.add(frame1, text = "FFM plot")
tabs.add(frame4, text = "GP plot")
tabs_number = 2;

tabs.pack(side = "left", anchor = "nw")



data_frame = Left_frame(frame0, win_width, win_height, dpi_all )

ffp = FFP_frame(frame1, win_width, win_height, dpi_all)

gp = GP_frame(frame4, win_width, win_height, dpi_all)

root.mainloop()