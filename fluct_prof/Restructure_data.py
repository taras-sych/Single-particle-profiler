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
		global file_index
		global rep_index

		temp_dict = {}

		for channel in range (data_cont.data_list_raw[file_index].datasets_list[rep_index].channels_number):

			x = []
			y = []

			temp_dict[channel] = fcs_importer.XY_plot(x,y)


		

			for rep_index_i in range (data_cont.data_list_raw[file_index].repetitions):
								
					

				#print ("adding repetition ", rep_index_i)


				if len(temp_dict[channel].x) == 0:
					x_min = 0
				else:
					x_min = max(temp_dict[channel].x) + temp_dict[channel].x[1] - temp_dict[channel].x[0]

				x_temp_1 = [elem + x_min for elem in data_cont.data_list_raw[file_index].datasets_list[rep_index_i].channels_list[channel].fluct_arr.x]

				temp_dict[channel].x.extend(x_temp_1)

				temp_dict[channel].y.extend(data_cont.data_list_raw[file_index].datasets_list[rep_index_i].channels_list[channel].fluct_arr.y)



		repetitions_new = int(self.num_rep.get())


		length_rep = int (len (temp_dict[0].x)/repetitions_new)
		

		
		dataset_list_arg = []
		for rep_index_i in range (repetitions_new):


			channels_list_arg = []

			for channel in range (data_cont.data_list_raw[file_index].datasets_list[rep_index].channels_number):

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

