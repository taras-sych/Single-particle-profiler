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

from fluct_prof import fcs_importer

from fluct_prof import Correlation as corr_py

from fluct_prof import Functions as fun

from fluct_prof import Data_container as data_cont

from fluct_prof import Data_tree as d_tree

#--------------------------
#End of importing own modules
#--------------------------









class Restruct_window:

	def Temp(self):
		print ("It is temp")

	def Restructure_dataset(self):


		temp_dict = {}

		for channel in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_number):

			x = []
			y = []

			temp_dict[channel] = fcs_importer.XY_plot(x,y)


		

			for rep_index_i in range (data_cont.data_list_raw[data_cont.file_index].repetitions):
								
					

				#print ("adding repetition ", rep_index_i)


				if len(temp_dict[channel].x) == 0:
					x_min = 0
				else:
					x_min = max(temp_dict[channel].x) + temp_dict[channel].x[1] - temp_dict[channel].x[0]

				x_temp_1 = [elem + x_min for elem in data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[channel].fluct_arr.x]

				temp_dict[channel].x.extend(x_temp_1)

				temp_dict[channel].y.extend(data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[channel].fluct_arr.y)



		repetitions_new = int(self.num_rep.get())


		length_rep = int (len (temp_dict[0].x)/repetitions_new)
		

		
		dataset_list_arg = []
		for rep_index_i in range (repetitions_new):


			channels_list_arg = []

			for channel in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_number):

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

				long_name = data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_list[channel].name

				short_name = data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_list[channel].short_name

				Ch_dataset = fcs_importer.fcs_channel (long_name, Tr, AutoCorr, short_name)

				channels_list_arg.append(Ch_dataset)

			FCS_Dataset =  fcs_importer.Dataset_fcs(data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_number, 0, channels_list_arg, [] )

			dataset_list_arg.append(FCS_Dataset)



		
		dataset = 	fcs_importer.Full_dataset_fcs(repetitions_new, dataset_list_arg)

		name = data_cont.tree_list_name[data_cont.file_index] + " " + str(repetitions_new)

		treetree = d_tree.Data_tree (self.tree, name, dataset.repetitions)

		treetree = d_tree.Data_tree (data_cont.data_frame.tree, name, dataset.repetitions)

		data_cont.tree_list.append(treetree)

		data_cont.tree_list_name.append(name)

		data_cont.binning_list.append(1)


		data_cont.data_list_raw.append(dataset)


		#data_list_current.append(dataset1)


		data_cont.total_channels_list.append(dataset.datasets_list[0].channels_number + dataset.datasets_list[0].cross_number)
		data_cont.repetitions_list.append(dataset.repetitions)

		data_cont.peaks_list.append([None] * dataset.repetitions)

		data_cont.list_of_channel_pairs.append([None])






	def Plot_curve(self):




		
		self.curves.cla()
		self.traces.cla()



		for item in data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_list:

			x1 = item.auto_corr_arr.x
			y1 = item.auto_corr_arr.y

		
			self.curves.plot(x1, y1, label = item.short_name)

		if data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].cross_number > 0:
			for item in data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].cross_list:

				x1 = item.cross_corr_arr.x
				y1 = item.cross_corr_arr.y

			
				self.curves.plot(x1, y1, label = item.short_name)



		
		
		self.curves.set_title("Correlation curves")
		self.curves.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
		self.curves.set_ylabel('G(tau)')
		self.curves.set_xlabel('Delay time')
		self.curves.set_xscale ('log')


		for item in data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_list:


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



		index = self.tree.selection()
		num1, num = index[0].split('I')


		

		num = int(num, 16)

		sum1 = num 
		file = 0

		rep = 0
		


		for i in range (len(data_cont.data_list_raw)):
			#print ("I am here")
			rep = 0
			sum1-=1
			file+=1
			if sum1 == 0:
				file1 = file
				rep1 = rep

			
			for j in range (data_cont.repetitions_list[i]):
				sum1-=1
				rep+=1
				if sum1 == 0:
					file1 = file
					rep1 = rep



		if rep1 == 0:
			rep1+=1




		

		data_cont.file_index = file1-1
		data_cont.rep_index = rep1-1



		self.Plot_curve()




	def __init__(self, win_width, win_height, dpi_all):




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
		self.num_rep.insert(0,data_cont.data_list_raw[data_cont.file_index].repetitions)

		self.Rep_button = tk.Button(self.frame004, text="Apply reps", command=self.Restructure_dataset)
		self.Rep_button.grid(row = 0, column = 2, rowspan = 2, sticky='w')

		Label_2 = tk.Label(self.frame004, text="Each rep: ")
		Label_2.grid(row = 1, column = 0, sticky = 'w')

		text1 = str(round(data_cont.data_list_raw[data_cont.file_index].datasets_list[data_cont.rep_index].channels_list[0].fluct_arr.x[-1],1)) + "sec"

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

		for i in range(0, len(data_cont.tree_list_name)):
			name = data_cont.tree_list_name[i]
			treetree = d_tree.Data_tree (self.tree, name, data_cont.data_list_raw[i].repetitions)

		self.tree.selection_set(treetree.child_id)

