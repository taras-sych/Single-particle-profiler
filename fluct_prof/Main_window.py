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

		
		
	def Continue_Import(self):
		print("Continuing import")
		self.dataset = fcs_importer.Fill_datasets_fcs(self.lines)
		"""		treetree = d_tree.Data_tree (self.tree, self.name, self.dataset.repetitions)
				self.tree.selection_set(treetree.child_id)
				data_cont.tree_list.append(treetree)

				data_cont.tree_list_name.append(self.name)

				data_cont.binning_list.append(1)


				data_cont.data_list_raw.append(self.dataset)


				#data_list_current.append(dataset1)


				data_cont.total_channels_list.append(self.dataset.datasets_list[0].channels_number + self.dataset.datasets_list[0].cross_number)
				data_cont.repetitions_list.append(self.dataset.repetitions)

				data_cont.peaks_list.append([None] * self.dataset.repetitions)

				data_cont.list_of_channel_pairs.append([None])"""

	def check_positions(self):

		for key in self.checklist.keys():
			if self.checklist[key].get() == 1:
				flag = 1
			else:
				flag = 0

			break


		if flag == 1:

			for key in self.checklist.keys():
				self.checklist[key].set(0)

		if flag == 0:

			for key in self.checklist.keys():
				self.checklist[key].set(1)


	def Import(self):

		


		

		

		if data_cont.initialdirectory == '':
			data_cont.initialdirectory = __file__

		ftypes = [('FCS .fcs', '*.fcs'), ('FCS .SIN', '*.SIN'), ('Text files', '*.txt'), ('All files', '*'), ]
		

		filenames =  tk.filedialog.askopenfilenames(initialdir=os.path.dirname(data_cont.initialdirectory),title = "Select file", filetypes = ftypes)

		
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

				data_cont.initialdirectory = os.path.dirname(filename)

				

				#progress_window.grab_set()


				self.name = os.path.basename(filename)

				file = codecs.open (filename, encoding='latin')

				self.lines = file.readlines()

				if filename.endswith('.fcs'):

					i = 0;

					while i < len(self.lines):



						if self.lines[i].__contains__("CarrierRows"):

							str1 , str2 = self.lines[i].split(' = ')
							CarrierRows = int(str2)

							str1 , str2 = self.lines[i+1].split(' = ')
							CarrierColumns = int(str2)

							print(CarrierRows, CarrierColumns)

							break

						i +=1

					self.checklist = {}

					check_button_list = {}

					labels_rows = [None] * CarrierRows
					labels_columns = [None] * CarrierColumns


					if CarrierColumns+CarrierRows > 2:

						self.win_check = tk.Toplevel()

						Label1 = tk.Label(self.win_check, text="Select cells to open: ")
						Label1.grid(row = 0, column = 0, columnspan = CarrierColumns+1, sticky='ew')

						for c in range (0,CarrierColumns):
							labels_columns[c] = tk.Label(self.win_check, text=str(c+1))
							labels_columns[c].grid(row = 1, column = c + 1, sticky='ew')

						for r in range (0,CarrierRows):
							labels_rows[r] = tk.Label(self.win_check, text=chr(r + 65))
							labels_rows[r].grid(row = r + 2, column = 0, sticky='ew')



						for r in range (0,CarrierRows):
							for c in range (0, CarrierColumns):
								self.checklist[r,c] = tk.IntVar(value = 1)

								check_button_list[r,c] = (tk.Checkbutton(self.win_check, variable=self.checklist[r,c]))
								check_button_list[r,c].grid(row = r + 2, column = c + 1, sticky='ew')
									

						
						Button_check_all = tk.Button(self.win_check, text="Check/uncheck all", command=self.check_positions)
						Button_check_all.grid(row = CarrierRows + 2, column = 0, columnspan = CarrierColumns+1, sticky='ew')

						Button_ok = tk.Button(self.win_check, text="OK", command=self.Continue_Import)
						Button_ok.grid(row = CarrierRows + 3, column = 0, columnspan = CarrierColumns+1, sticky='ew')
					
					else:self.Continue_Import()


				if filename.endswith('.SIN'): 
					self.dataset = fcs_importer.Fill_datasets_sin(lines)

				#dataset1 = copy.deepcopy(dataset)


				

				#root.update() 



		self.pb.destroy()
		self.value_label.destroy()



	def Select_Unselect(self):

		

		self.Plot_this_data(data_cont.data_list_raw[data_cont.file_index], data_cont.rep_index)

		data_cont.root.update()


	def Plot_data(self, event):

		start = time.time()

		

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


		

		

		rep = rep1-1


		self.Curve_flags()

		

		self.Plot_this_data(data_cont.data_list_raw[data_cont.file_index], rep)

		#root.update()

	def Delete_dataset(self):
		
		index = self.tree.selection()
		for sel in index:
			self.tree.delete(sel)

	def Delete_all_datasets(self):
		



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

		for i in range (len(data_cont.data_list_raw)):
			if data_cont.data_list_raw[i].datasets_list[0].channels_number > channels_to_display:
				channels_to_display = data_cont.data_list_raw[i].datasets_list[0].channels_number
				file_index_local = i


		for item in data_cont.data_list_raw[file_index_local].datasets_list[data_cont.rep_index].channels_list:
			str1, str2 = item.short_name.split(" ")
			very_short_name = "ch0" + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command = self.Select_Unselect)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1

		for item in data_cont.data_list_raw[file_index_local].datasets_list[data_cont.rep_index].cross_list:
			str1, str2 = item.short_name.split(" vs ")
			str3, str4 = str1.split(" ")
			very_short_name = "ch" + str4 + str2
			self.channels_flags[item.short_name] = tk.IntVar(value=1)
			self.flags_dict[item.short_name] = tk.Checkbutton(self.frame0003, text=very_short_name, variable=self.channels_flags[item.short_name], command = self.Select_Unselect)
			self.flags_dict[item.short_name].grid(row = 0, column = column_counter, sticky='w')
			column_counter +=1


	def __init__ (self, frame0, win_width, win_height, dpi_all):



		pixel = tk.PhotoImage(width=1, height=1)


		

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


		self.Restruct_button = tk.Button(self.frame023, text="Restructure data", command=fun.Restruct_fun)
		self.Restruct_button.grid(row = 0, column = 0, sticky="EW")

		self.Threshold_button = tk.Button(self.frame023, text="Peak analysis", command=fun.Threshold_fun)
		self.Threshold_button.grid(row = 1, column = 0, sticky="EW")

		self.Diffusion_button = tk.Button(self.frame023, text="Diffusion analysis", command=fun.Diffusion_fun)
		self.Diffusion_button.grid(row = 2, column = 0, sticky="EW")

		self.Add_to_plot_button = tk.Button(self.frame023, text="Plot", command=fun.Which_tab)
		self.Add_to_plot_button.grid(row = 3, column = 0, sticky="EW")

		
		self.Add_to_plot_button = tk.Button(self.frame023, text="Dot Plot", command=fun.Dot_Plot_fun)
		self.Add_to_plot_button.grid(row = 4, column = 0, sticky="EW")

		self.Output_button = tk.Button(self.frame023, text="Output", command=fun.Export_function)
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