#--------------------------
#Importing general modules
#--------------------------
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import matplotlib.pyplot as plt

from scipy.stats import kde

import csv

import lmfit

import time

import umap


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

import pandas as pd


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

def Norm():
	print("This is a temporary function")



class UMAP_Window:

	def Save_plot_data(self):

		for key in self.thisdict_axis_1.keys():


			filename = data_cont.initialdirectory + "\\" +  key + "_Dot_Plot.txt"

			open_file = open(filename, 'w')


			open_file.write(str(key) + "\n")
			open_file.write(str(self.string_x) + "\t" + str(self.string_y) + "\n")

			for i in range(len(self.thisdict_axis_1[key])):
				open_file.write(str(self.thisdict_axis_1[key][i]) + "\t" + str(self.thisdict_axis_2[key][i]) + "\n")

			open_file.close()



	def Choose_dataset(self, event):




		index = self.tree.selection()

		num1, num = index[0].split('I')
		

		num = int(num, 16)

		

		sum1 = num 
		file = 0
		rep = 0
		for i in range (len(data_cont.data_list_raw)):
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


		


		output_file_name = data_cont.tree_list_name[file1-1][:-4]




		file1 = file1-1
		rep1 = rep1-1




		output_file_name = data_cont.tree_list_name[file1-1][:-4]




		data_cont.file_index = file1-1
		data_cont.rep_index = rep1-1

		self.axis_choice = []


		

		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
				
				str1 = data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[i].short_name
				self.axis_choice.append(str1)
				self.axis_choice.append("prom_" + str1)
				self.axis_choice.append("width_" + str1)


		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
				
				str1 = "Diff_" + data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[i].short_name
				self.axis_choice.append(str1)

		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].cross_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].cross_number):
				
				str1 = "Diff_" + data_cont.data_list_raw[data_cont.file_index].datasets_list[0].cross_list[i].short_name
				self.axis_choice.append(str1)

		self.axis_choice.append("GP")


		self.Axis_y_label__choice.config(values = self.axis_choice)
		self.Axis_x_label__choice.config(values = self.axis_choice)




		




	def Plot_dataset(self):





		
		#self.dens_plot.cla()
		#self.colorbar.cla()



		list1 = self.tree.get_checked()

		
		file2 = -1


		self.thisdict_axis_1 = {}
		self.thisdict_axis_2 = {}

		for index in list1:

			num1, num = index.split('I')
			

			num = int(num, 16)

			

			sum1 = num 
			file = 0
			rep = 0
			for i in range (len(data_cont.data_list_raw)):
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


			
			

			#output_file_name = data_cont.tree_list_name[file1-1][:-4]



			#comment123
			file1 = file1-1
			rep1 = rep1-1

			if file1 > file2:

				file2 = file1


				dimension = data_cont.data_list_raw[file1].export_dataframe["Intensity peaks"].shape[0]

				name = data_cont.tree_list_name[file1]

				list1 = [name] * dimension


				df = pd.concat([pd.DataFrame({"Compositions": list1}), data_cont.data_list_raw[file1].export_dataframe["Intensity peaks"]], axis = 1)




				try:
					merged_df = pd.concat([merged_df, df])

				except:
					merged_df = df


		


		try:
			merged_df = merged_df.drop('time', axis=1)
		except:
			pass

		



		filename = data_cont.initialdirectory + os.sep +  "Intensities.xlsx"

		print(filename)

		open_file = open(filename, 'w')

		

		



		writer = pd.ExcelWriter(filename, engine='xlsxwriter')

		#for key in self.data_frames_import.keys():
		merged_df.to_excel(writer, sheet_name="Intensities")

		writer.close()


		if self.norm_variable.get() == 1:

			merged_df = self.normalize_data_channels(merged_df, self.Norm_list.get())

			print (merged_df)

		self.do_umap(merged_df)


	
	def normalize_data_channels(self, df, normalization):

		if normalization == "max":


			convert_dict = {} 
			for key in list(df.columns[1:]):
				convert_dict [key] = float

			df = df.astype(convert_dict)

			max_values = df.iloc[:, 1:].max(axis=1)



			for index in range(len(df)):
				max_value = df.iloc[index, 1:].max()
				df.iloc[index, 1:] = df.iloc[index, 1:].div(max_value)


			return df


		if normalization != "max":

			convert_dict = {} 
			for key in list(df.columns[1:]):
				convert_dict [key] = float

			df = df.astype(convert_dict)

						
			divisor_column = normalization

			print(divisor_column)
			print(df.columns)

			
			for index in range(len(df)):
				divisor_value = df.at[index, divisor_column]
				#if divisor_value!= 0:
				df.iloc[index, 1:] = df.iloc[index, 1:].div(divisor_value)

			return df




	def do_umap(self, df):

		#Extract the labels from the 'Compositions' column as a Categorical series with ordered=True
		labels = df['Compositions'].values.tolist()
		label_categories = pd.Categorical(labels, categories=pd.unique(labels), ordered=True)

		# Remove the 'Compositions' column and convert the data to a NumPy array
		data = df.drop('Compositions', axis=1).values

		# Perform UMAP on the data to reduce the dimensionality
		umap_reducer = umap.UMAP(n_components=2, random_state=45)
		transformed_data = umap_reducer.fit_transform(data)

		# Create a dictionary that maps unique labels to unique colors
		unique_labels = label_categories.categories.tolist()
		num_labels = len(unique_labels)
		color_map = dict(zip(unique_labels, np.linspace(0, 1, num_labels)))

		# Convert the labels to colors using the color map
		colors = [color_map[label] for label in label_categories]

		# Add number of data points for each specific label in the legend
		label_counts = df['Compositions'].value_counts()
		handles = []
		for label in unique_labels:
		    color = color_map[label]
		    count = label_counts[label]
		    handle = plt.plot([], [], marker="o", markersize=25, ls="", mec=None, mew=0, color=plt.cm.tab10(color),
		                      label=f'{label} (n={count})')
		    handles.append(handle[0])

		# Plot the transformed data in 2D using the top two UMAP components
		#fig, ax1 = plt.subplots(figsize=(20, 11.25))
		#fig.subplots_adjust(left=0.05, right=0.7, bottom=0.05, top=0.95)

		key = self.Plot_list.get()

		colormap = self.Cmap_list.get()
		transparency = float(self.Transparency_entry.get())

		self.dot_plot[key].scatter(transformed_data[:, 0], transformed_data[:, 1], c=colors, cmap=colormap, alpha=transparency)

		# Add legend to the plot
		self.dot_plot[key].legend(handles=handles, loc='upper right')

		# Remove the axis labels
		self.dot_plot[key].set_xticks([])
		self.dot_plot[key].set_yticks([])


		self.canvas5.draw_idle()

		self.figure5.tight_layout()



			




			







		
			
		

	def __init__(self, win_width, win_height, dpi_all):

		self.channel_index = 0
		self.fit_all_flag = False


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

		#self.tree.bind('<<TreeviewSelect>>', self.Choose_dataset)



		self.Datalist.config(width = 100, height = 10)

		for i in range(0, len(data_cont.tree_list_name)):
			name = data_cont.tree_list_name[i]
			treetree = d_tree.Data_tree (self.tree, name, data_cont.data_list_raw[i].repetitions)


		self.frame003 = tk.Frame(self.frame002)
		self.frame003.pack(side = "top", anchor = "nw")

		self.frame0003 = tk.Frame(self.frame003)
		self.frame0003.pack(side = "top", anchor = "nw")



		self.frame001 = tk.Frame(self.frame002)
		self.frame001.pack(side = "top", anchor = "nw")

		self.frame000 = tk.Frame(self.win_dot_plot)
		self.frame000.pack(side = "left", anchor = "nw")


		self.figure5 = Figure(figsize=(0.9*self.th_width/dpi_all,0.9*self.th_height/(dpi_all)), dpi = dpi_all)
						
		gs = self.figure5.add_gridspec(2, 2)


		self.dot_plot = {}

		self.dot_plot["Plot 1"] = self.figure5.add_subplot(gs[0, 0])

		self.dot_plot["Plot 1"].set_title("Dot Plot 1")


		self.dot_plot["Plot 1"].set_ylabel('axis 2')
		self.dot_plot["Plot 1"].set_xlabel('axis 1')


		self.dot_plot["Plot 2"] = self.figure5.add_subplot(gs[0, 1])

		self.dot_plot["Plot 2"].set_title("Dot Plot 2")


		self.dot_plot["Plot 2"].set_ylabel('axis 2')
		self.dot_plot["Plot 2"].set_xlabel('axis 1')

		self.dot_plot["Plot 3"] = self.figure5.add_subplot(gs[1, 0])

		self.dot_plot["Plot 3"].set_title("Dot Plot 3")


		self.dot_plot["Plot 3"].set_ylabel('axis 2')
		self.dot_plot["Plot 3"].set_xlabel('axis 1')

		self.dot_plot["Plot 4"] = self.figure5.add_subplot(gs[1, 1])

		self.dot_plot["Plot 4"].set_title("Dot Plot 4")


		self.dot_plot["Plot 4"].set_ylabel('axis 2')
		self.dot_plot["Plot 4"].set_xlabel('axis 1')

		
		



		self.canvas5 = FigureCanvasTkAgg(self.figure5, self.frame000)
		self.canvas5.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

		self.toolbar = NavigationToolbar2Tk(self.canvas5, self.frame000)
		self.toolbar.update()
		self.canvas5.get_tk_widget().pack()

		self.figure5.tight_layout()

		self.Export_plot_button = tk.Button(self.frame000, text="Save plot data", command=self.Save_plot_data)
		self.Export_plot_button.pack(side = "top", anchor = "nw")

		

		

		self.Plot_label = tk.Label(self.frame001, text = "Plot on: ")
		self.Plot_label.grid(row = 0, column = 0, sticky = 'w')

		self.Plot_list = ttk.Combobox(self.frame001,values = ["Plot 1", "Plot 2", "Plot 3", "Plot 4"],  width = 18)
		self.Plot_list.config(state = "readonly")
		self.Plot_list.set("Plot 1")

		self.Plot_list.grid(row = 0, column = 1)

		self.Norm_label = tk.Label(self.frame001, text = "Normalize: ")
		self.Norm_label.grid(row = 1, column = 0, sticky = 'w')

		self.Norm_list = ttk.Combobox(self.frame001, values = ["max", "channel 1", "channel 2", "channel 3", "channel 4", "channel 5", "channel 6"],  width = 18)
		self.Norm_list.config(state = "readonly")
		self.Norm_list.set("max")

		self.Norm_list.grid(row = 1, column = 1)

		self.norm_variable = tk.IntVar()

		self.Norm_check = tk.Checkbutton(self.frame001, text="Normalize", variable = self.norm_variable)
		self.Norm_check.grid(row = 2, column = 0, columnspan = 2, sticky = 'w')

		self.Transparency_label = tk.Label(self.frame001, text = "Transparency: ")
		self.Transparency_label.grid(row = 3, column = 0, sticky = 'w')

		self.Transparency_entry = tk.Entry(self.frame001, width = 18)
		self.Transparency_entry.grid(row = 3, column = 1, sticky = 'w')

		self.Cmap_label = tk.Label(self.frame001, text = "Colormap: ")
		self.Cmap_label.grid(row = 4, column = 0, sticky = 'w')

		self.Cmap_list = ttk.Combobox(self.frame001, values = plt.colormaps(),  width = 18)
		self.Cmap_list.config(state = "readonly")
		self.Cmap_list.set("viridis")

		self.Cmap_list.grid(row = 4, column = 1)



		self.Plot_button = tk.Button(self.frame001, text="Cenk does magic", command = self.Plot_dataset)
		self.Plot_button.grid(row = 5, column = 0, columnspan = 2, sticky = 'ew')




		
		self.tree.selection_set(treetree.child_id)