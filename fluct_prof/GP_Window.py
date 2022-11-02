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

from scipy.signal import find_peaks, peak_widths, peak_prominences

from scipy.optimize import curve_fit
import random

import seaborn as sns


#--------------------------
#End of importing general modules
#--------------------------


#--------------------------
#Importing own modules
#--------------------------

from fluct_prof import Functions as fun

from fluct_prof import Data_container as data_cont

from fluct_prof import Data_tree as d_tree

from fluct_prof import fcs_importer

#--------------------------
#End of importing own modules
#--------------------------














class Threshold_window:

	def Save_plot_data(self):

		print(self.save_plot_dict.keys())



		name = data_cont.tree_list_name[data_cont.file_index]
		filename = data_cont.initialdirectory + "\\" +  name + "_Plots_gp.txt"

		open_file = open(filename, 'w')

		for key in self.save_plot_dict.keys():

			if key.__contains__("channel") == True and key.__contains__("fluct") == True and self.plot_var["Traces"].get() == 1:

			#print(key)
				open_file.write(str(key) + "\n")

				for i in range(len(self.save_plot_dict[key].x)):
					open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

				open_file.write("\n")
				open_file.write("\n")



			if key.__contains__("channel") == True and key.__contains__("peaks") == True and self.plot_var["Peaks"].get() == 1:

				str1, str2 = key.split("p")
				key1 = str1 + "prominences"

				print (key1)

				str1, str2 = key.split("p")
				key2 = str1 + "widths"

				print (key2)

				open_file.write(str(key) + "\t" + str(key1) + "\t" + str(key2))

				for i in range(len(self.save_plot_dict[key].x)):
					open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) +  "\t" + str(self.save_plot_dict[key1].y[i]) +  "\t" + str(self.save_plot_dict[key2].y[i]) + "\n")

				open_file.write("\n")
				open_file.write("\n")


			if key.__contains__("gp") == True and self.plot_var["GP Plot"].get() == 1:

				open_file.write(str(key) + "\n")

				for i in range(len(self.save_plot_dict[key].x)):
					open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

				open_file.write("\n")
				open_file.write("\n")

			if key.__contains__("sum") == True and self.plot_var["GP Fit"].get() == 1:

				open_file.write(str(key) + "\n")

				for i in range(len(self.save_plot_dict[key].x)):
					open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

				open_file.write("\n")
				open_file.write("\n")


			if key.__contains__("component") == True and self.plot_var["GP Fit"].get() == 1:

				open_file.write(str(key) + "\n")

				for i in range(len(self.save_plot_dict[key].x)):
					open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

				open_file.write("\n")
				open_file.write("\n")


			if key.__contains__("dot plot") == True:

				open_file.write(str(key) + "\n")

				for i in range(len(self.save_plot_dict[key].x)):
					open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

				open_file.write("\n")
				open_file.write("\n")

			if key.__contains__("channel") == True and key.__contains__("hist") == True:

				open_file.write(str(key) + "\n")

				for i in range(len(self.save_plot_dict[key].x)):
					open_file.write(str(self.save_plot_dict[key].x[i]) + "\t" + str(self.save_plot_dict[key].y[i]) + "\n")

				open_file.write("\n")
				open_file.write("\n")





		open_file.close()

	def Apply_to_all(self):

		

		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()

		for rep_index_i in range (data_cont.data_list_raw[data_cont.file_index].repetitions):
			for param in self.list_of_params:
				self.full_dict[param]["Init"].delete(0,"end")
				self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))
				
			data_cont.rep_index = rep_index_i
			#self.Normalize()
			self.Peaks()
			self.Fit_gaus()

		self.fit_all_flag = False

		self.Peaks()



	def Apply_to_all_ticks(self):



		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()



		list1 = self.tree_t.get_checked()



		thisdict = {}

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



			data_cont.file_index = file1-1
			data_cont.rep_index = rep1-1




			for param in self.list_of_params:
				self.full_dict[param]["Init"].delete(0,"end")
				self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))
				
			
			#self.Normalize()
			self.Peaks()
			self.Fit_gaus()

		self.fit_all_flag = False

		self.Peaks()


	def Apply_to_all_all(self):



		self.fit_all_flag = True

		self.list_of_inits_for_fit_all = {}

		for param in self.list_of_params:
			self.list_of_inits_for_fit_all[param] = self.full_dict[param]["Init"].get()

		for file_index_i in range (len(data_cont.data_list_raw)):	
			for rep_index_i in range (data_cont.data_list_raw[file_index_i].repetitions):
				for param in self.list_of_params:
					self.full_dict[param]["Init"].delete(0,"end")
					self.full_dict[param]["Init"].insert(0,str(round(float(self.list_of_inits_for_fit_all[param]),3)))
					
				data_cont.rep_index = rep_index_i
				data_cont.file_index = file_index_i

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
			y_model = fun.Gauss(x, *param_list)

		if self.Components.get() == '2 components':
			y_model = fun.Gauss2(x, *param_list)

		if self.Components.get() == '3 components':
			y_model = fun.Gauss3(x, *param_list)
		return y_model - ydata


	def Fit_gaus(self):

		
		

			
		


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


		data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] = output_dict

		


			





			
		if self.fit_all_flag == False:
			self.gp_hist.cla()
										
										
			self.gp_hist.set_title("GP histogram")
			self.gp_hist.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
			self.gp_hist.set_ylabel('Counts (Total: ' + str(sum(self.n)) + ')' )
			self.gp_hist.set_xlabel('GP')
			self.gp_hist.bar(x, y, width = x[1] - x[0], bottom=None, align='center', label = 'raw')
			

			if self.Components.get() == '1 component':
				self.gp_hist.plot(x1, fun.Gauss(x1, *popt), 'r-', label='fit')
				self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt))

			if self.Components.get() == '2 components':
				self.gp_hist.plot(x1, fun.Gauss2(x1, *popt), 'r-', label='fit')
				self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, fun.Gauss2(x1, *popt))
				popt1 = popt[:3]
				popt2 = popt[3:6]
				self.gp_hist.plot(x1, fun.Gauss(x1, *popt1), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, fun.Gauss(x1, *popt2), color = 'yellow', label='fit')

				self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt1))
				self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt2))

			if self.Components.get() == '3 components':
				self.gp_hist.plot(x1, fun.Gauss3(x1, *popt), 'r-', label='fit')
				self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, fun.Gauss3(x1, *popt))
				popt1 = popt[:3]
				popt2 = popt[3:6]
				popt3 = popt[6:9]
				self.gp_hist.plot(x1, fun.Gauss(x1, *popt1), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, fun.Gauss(x1, *popt2), color = 'yellow', label='fit')
				self.gp_hist.plot(x1, fun.Gauss(x1, *popt3), color = 'yellow', label='fit')

				self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt1))
				self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt2))
				self.save_plot_dict["component 3"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt3))



			self.canvas5.draw_idle()

			self.figure5.tight_layout()

	
	def Update_thresholds (self):


		data_cont.change_normal = False

		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number == 1:

			data_cont.data_list_raw[data_cont.file_index].threshold_list[0] = float(self.ch1_th.get())
		else:





			self.Channel_pair__choice.config(values = self.channel_pairs)
			self.Channel_pair__choice.set(self.channel_pairs[0])



			if data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] != None:
				data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] = None





			data_cont.data_list_raw[data_cont.file_index].threshold_list[0] = float(self.ch1_th.get())
			data_cont.data_list_raw[data_cont.file_index].threshold_list[1] = float(self.ch2_th.get())



			




		self.Peaks()




	def Peaks (self):

		self.save_plot_dict = {}


		if self.fit_all_flag == False:
			self.peaks.cla()
			self.hist1.cla()
			self.gp_hist.cla()

			self.canvas5.draw_idle()

			self.figure5.tight_layout()

		

		


		
		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number == 1:

			ch1_ind = 0

			main_xlim = self.peaks.get_xlim()
			main_ylim = self.peaks.get_ylim()

			int_div = int(data_cont.rep_index/data_cont.data_list_raw[data_cont.file_index].binning)

			x1 = []
			
			y1 = []
			
			y1_raw = []

			for rep_index_i in range (data_cont.data_list_raw[data_cont.file_index].repetitions):
							
				if int(rep_index_i/data_cont.data_list_raw[data_cont.file_index].binning) == int_div:

					


					if len(x1) == 0:
						x_min = 0
					else:
						x_min = max(x1) + x1[1] - x1[0]

					x_temp_1 = [elem + x_min for elem in data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[ch1_ind].fluct_arr.x]

					x1.extend(x_temp_1)
					#y1.extend(data_list_current[data_cont.file_index].datasets_list[rep_index_i].channels_list[0].fluct_arr.y)
					y1_raw.extend(data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[ch1_ind].fluct_arr.y)

			th1 = data_cont.data_list_raw[data_cont.file_index].threshold_list[ch1_ind]

			if th1 == None:

				if self.normalization_index == "z-score":

					th1 = 2
					

				if self.normalization_index == "manual":


					th1 = 2
					

			self.ch1_th.delete(0,"end")
			self.ch1_th.insert(0,str(th1))


			if self.normalization_index == "z-score":
				y1 = stats.zscore(y1_raw)
		

			if self.normalization_index == "manual":


				y1 = y1_raw/np.mean(y1_raw)

			

			yh1 = []




			
			

			for el in y1:
				if el >= th1:
					yh1.append(el)





			#which_channel = self.Threshold.get()

			
			peaks, _ = find_peaks(y1, height=th1)




			xp1 = []

			yp1 = []

			yp1_raw = []

			yp1_raw_sep = []





			for p in peaks:
				yp1_raw_sep.append(y1_raw[p])




			for p in peaks:
				xp1.append(x1[p])

				yp1.append(y1[p])


				yp1_raw.append(y1_raw[p])




			
			

			if self.fit_all_flag == False:
				self.peaks.cla()
				self.hist1.cla()
				self.gp_hist.cla()
				self.dot_plot.cla()

				self.peaks.set_title("Intensity traces")
				
				self.peaks.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
				self.peaks.set_ylabel('Intensity (a.u.)')
				self.peaks.set_xlabel('Time (s)')

				self.hist1.set_title("Peak intensity histograms")

				self.hist1.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
				self.hist1.set_ylabel('Counts')
				self.hist1.set_xlabel('Intensity (a.u.)')




				self.peaks.plot(x1, y1_raw, '#1f77b4', zorder=1)
				#self.peaks.hlines(th1, min(x1), max(x1), color = 'magenta', zorder=2)
				
				if (self.var.get() == 1):
					self.peaks.plot(xp1, yp1_raw, "x", color = 'magenta', zorder = 3)

				bins_1 = int(np.sqrt(len(yh1)))
				if bins_1 == 0:
					bins_1 = 1
				self.hist1.hist(yp1_raw_sep, bins = bins_1, label = "total: " + str(len(yp1_raw_sep)))

				self.save_plot_dict["channel 1 fluct"] = fcs_importer.XY_plot(x1, y1_raw)
				self.save_plot_dict["channel 1 peaks"] = fcs_importer.XY_plot(xp1, yp1_raw)
					



				self.hist1.legend(loc='upper right')

				self.canvas5.draw_idle()

				self.figure5.tight_layout()
					
			



		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number > 1:




			str1 = self.Channel_pair__choice.get()

			str7, str2 = str1.split('/')

			str3, str4 = str7.split(' ')

			str5, str6 = str2.split(' ')

			ch1_ind = int(str4) - 1

			ch2_ind = int(str6) - 1



			main_xlim = self.peaks.get_xlim()
			main_ylim = self.peaks.get_ylim()




			int_div = int(data_cont.rep_index/data_cont.data_list_raw[data_cont.file_index].binning)

			


			x1 = []
			x2 = []
			y1 = []
			y2 = []
			y1_raw = []
			y2_raw = []


			



			for rep_index_i in range (data_cont.data_list_raw[data_cont.file_index].repetitions):
							
				if int(rep_index_i/data_cont.data_list_raw[data_cont.file_index].binning) == int_div:

					


					if len(x1) == 0:
						x_min = 0
					else:
						x_min = max(x1) + x1[1] - x1[0]

					x_temp_1 = [elem + x_min for elem in data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[ch1_ind].fluct_arr.x]
					x_temp_2 = [elem + x_min for elem in data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[ch2_ind].fluct_arr.x]


					x1.extend(x_temp_1)
					#y1.extend(data_list_current[data_cont.file_index].datasets_list[rep_index_i].channels_list[0].fluct_arr.y)
					y1_raw.extend(data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[ch1_ind].fluct_arr.y)

					x2.extend(x_temp_2)
					#y2.extend(data_list_current[data_cont.file_index].datasets_list[rep_index_i].channels_list[1].fluct_arr.y)
					y2_raw.extend(data_cont.data_list_raw[data_cont.file_index].datasets_list[rep_index_i].channels_list[ch2_ind].fluct_arr.y)




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



			th1 = data_cont.data_list_raw[data_cont.file_index].threshold_list[ch1_ind]
			th2 = data_cont.data_list_raw[data_cont.file_index].threshold_list[ch2_ind]

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

			widths1 = peak_widths(y1, peaks, rel_height=0.5)

			widths2 = peak_widths(y2, peaks, rel_height=0.5)

			prominences1 = peak_prominences(y1, peaks)[0]

			prominences2 = peak_prominences(y2, peaks)[0]

			xp1 = []
			xp2 = []
			yp1 = []

			yp1_raw = []
			yp2 = []

			yp2_raw = []

			yp1_raw_sep = []
			yp2_raw_sep = []




			for p in peaks1:
				yp1_raw_sep.append(y1_raw[p])

			for p in peaks2:
				yp2_raw_sep.append(y2_raw[p])


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

				self.hist1.set_title("Peak intensity histograms")

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


					if self.Normalization_for_plot.get() == "Peak Intensity": 

						self.hist1.set_title("Peak intensity histograms")

						self.n, bins, patches = self.hist1.hist(yp1_raw, bins = bins_1, label = "total: " + str(len(yp1_raw)))

					if self.Normalization_for_plot.get() == "Peak Prominence":

						self.hist1.set_title("Peak prominence histograms")

						self.n, bins, patches = self.hist1.hist(prominences1, bins = bins_1, label = "total: " + str(len(prominences1)))

					if self.Normalization_for_plot.get() == "Peak width at half max":

						self.hist1.set_title("Peak width histograms")

						self.n, bins, patches = self.hist1.hist(widths1, bins = bins_1, label = "total: " + str(len(widths1)))

					self.save_plot_dict["channel 1 fluct"] = fcs_importer.XY_plot(x1, y1_raw)
					self.save_plot_dict["channel 1 peaks"] = fcs_importer.XY_plot(xp1, yp1_raw)
					self.save_plot_dict["channel 1 prominences"] = fcs_importer.XY_plot(xp1, prominences1)
					self.save_plot_dict["channel 1 widths"] = fcs_importer.XY_plot(xp1, widths1)

					self.x_bins=[]
					for ii in range (len(bins)-1):
						self.x_bins.append( (bins[ii+1] - bins[ii])/2 + bins[ii])

					self.save_plot_dict["channel 1 hist"] = fcs_importer.XY_plot(self.x_bins, self.n)
					

				if which_channel == "channel 2" or which_channel == "both or" or which_channel == "both and":
					
					self.peaks.plot(x2, y2_raw, '#ff7f0e', zorder=1)
					#self.peaks.hlines(th2, min(x2), max(x2), color = 'green', zorder=2)

					if (self.var.get() == 1):
						self.peaks.plot(xp2, yp2_raw, "x", color = 'green', zorder = 3)

					bins_2 = int(np.sqrt(len(yh2)))
					if bins_2 == 0:
						bins_2 = 1
					

					if self.Normalization_for_plot.get() == "Peak Intensity":

						self.hist1.set_title("Peak intensity histograms")

						self.n, bins, patches = self.hist1.hist(yp2_raw, bins = bins_2, label = "total: " + str(len(yp2_raw)))

					if self.Normalization_for_plot.get() == "Peak Prominence":

						self.hist1.set_title("Peak prominence histograms")

						self.n, bins, patches = self.hist1.hist(prominences2, bins = bins_2, label = "total: " + str(len(prominences2)))

					if self.Normalization_for_plot.get() == "Peak width at half max":

						self.hist1.set_title("Peak width histograms")

						self.n, bins, patches = self.hist1.hist(widths2, bins = bins_2, label = "total: " + str(len(widths2)))



					self.save_plot_dict["channel 2 fluct"] = fcs_importer.XY_plot(x2, y2)
					self.save_plot_dict["channel 2 peaks"] = fcs_importer.XY_plot(xp2, yp2)
					self.save_plot_dict["channel 2 prominences"] = fcs_importer.XY_plot(xp2, prominences2)
					self.save_plot_dict["channel 2 widths"] = fcs_importer.XY_plot(xp2, widths2)

					self.x_bins=[]
					for ii in range (len(bins)-1):
						self.x_bins.append( (bins[ii+1] - bins[ii])/2 + bins[ii])

					self.save_plot_dict["channel 2 hist"] = fcs_importer.XY_plot(self.x_bins, self.n)

				"""if data_cont.change_normal == False:
														self.peaks.set_xlim(main_xlim)
														self.peaks.set_ylim(main_ylim)"""

				self.hist1.legend(loc='upper right')

			gp_list_temp = []
			peaks_y_temp = []
			peaks_x_temp = []



			

			
			for k in range (len(yp1_raw)):
				gp_1 = (yp1_raw[k] - yp2_raw[k])/(yp2_raw[k] + yp1_raw[k])

				peaks_x_temp.append(yp1_raw[k])
				peaks_y_temp.append(yp2_raw[k])


				if abs(gp_1) < 1:
					gp_list_temp.append(gp_1)




			data_cont.data_list_raw[data_cont.file_index].peaks[data_cont.rep_index, ch1_ind] = peaks_x_temp
			data_cont.data_list_raw[data_cont.file_index].peaks[data_cont.rep_index, ch2_ind] = peaks_y_temp

			data_cont.data_list_raw[data_cont.file_index].peak_prominences[data_cont.rep_index, ch1_ind] = prominences1
			data_cont.data_list_raw[data_cont.file_index].peak_prominences[data_cont.rep_index, ch2_ind] = prominences2

			data_cont.data_list_raw[data_cont.file_index].peak_widths[data_cont.rep_index, ch1_ind] = widths1
			data_cont.data_list_raw[data_cont.file_index].peak_widths[data_cont.rep_index, ch2_ind] = widths2

			if self.Normalization_for_plot.get() == "Peak Intensity":

				axis_x_temp = peaks_x_temp
				axis_y_temp = peaks_y_temp


			if self.Normalization_for_plot.get() == "Peak Prominence":

				axis_x_temp = prominences1
				axis_y_temp = prominences2

			if self.Normalization_for_plot.get() == "Peak width at half max":

				axis_x_temp = widths1
				axis_y_temp = widths2


			
			self.n, bins, patches = self.gp_hist.hist(gp_list_temp, bins = int(np.sqrt(len(gp_list_temp))))

			self.dot_plot.scatter(axis_x_temp, axis_y_temp)
			self.dot_plot.ticklabel_format(axis = "x", style="sci", scilimits = (0,0))
			self.dot_plot.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))


			self.save_plot_dict["dot plot"] = fcs_importer.XY_plot(axis_x_temp, axis_y_temp)

				
			

			self.x_bins=[]
			for ii in range (len(bins)-1):
				self.x_bins.append( (bins[ii+1] - bins[ii])/2 + bins[ii])

			self.save_plot_dict["gp histogram"] = fcs_importer.XY_plot(self.x_bins, self.n)


			if self.fit_all_flag == False:
				self.gp_hist.set_title("GP histogram")
				self.gp_hist.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
				self.gp_hist.set_ylabel('Counts (Total: ' + str(len(gp_list_temp)) + ')' )
				self.gp_hist.set_xlabel('GP')


				if data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] != None:


					x1 = np.linspace(min(self.x_bins), max(self.x_bins), num=500)
					popt = []

					for param in data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index].keys():
				

						popt.append(np.float64(data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index][param]))




					if self.Components.get() == '1 component':
						#print("1 comp")
						self.gp_hist.plot(x1, fun.Gauss(x1, *popt), 'r-', label='fit')

						self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt))
						

					if self.Components.get() == '2 components':
						#print("2 comp")
						self.gp_hist.plot(x1, fun.Gauss2(x1, *popt), 'r-', label='fit')
						self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, fun.Gauss2(x1, *popt))

						popt1 = popt[:3]
						popt2 = popt[3:6]
						
						self.gp_hist.plot(x1, fun.Gauss(x1, *popt1), color = 'yellow', label='fit')
						self.gp_hist.plot(x1, fun.Gauss(x1, *popt2), color = 'yellow', label='fit')

						self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt1))
						self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt2))

					if self.Components.get() == '3 components':
						self.gp_hist.plot(x1, fun.Gauss3(x1, *popt), 'r-', label='fit')
						self.save_plot_dict["sum of gaussians"] = fcs_importer.XY_plot(x1, fun.Gauss3(x1, *popt))
						#print("3 comp")
						popt1 = popt[:3]
						popt2 = popt[3:6]
						popt3 = popt[6:9]
						
						self.gp_hist.plot(x1, fun.Gauss(x1, *popt1), color = 'yellow', label='fit')
						self.gp_hist.plot(x1, fun.Gauss(x1, *popt2), color = 'yellow', label='fit')
						self.gp_hist.plot(x1, fun.Gauss(x1, *popt3), color = 'yellow', label='fit')

						self.save_plot_dict["component 1"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt1))
						self.save_plot_dict["component 2"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt2))
						self.save_plot_dict["component 3"] = fcs_importer.XY_plot(x1, fun.Gauss(x1, *popt3))





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

			if data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] != None and len(data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index].keys()) == len(self.list_of_params) :

				thisdict["Init"].delete(0,"end")
				thisdict["Init"].insert(0,data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index][param])



			



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



		


		

		for rep in range(data_cont.repetitions_list[data_cont.file_index]):


			y1 = data_cont.data_list_raw[data_cont.file_index].datasets_list[rep].channels_list[0].fluct_arr.y
			y2 = data_cont.data_list_raw[data_cont.file_index].datasets_list[rep].channels_list[1].fluct_arr.y
			
				
			if self.normalization_index == "z-score":
				y1z = stats.zscore(y1)
				y2z = stats.zscore(y2)


				data_cont.data_list_raw[data_cont.file_index].threshold_list[0] = float(self.ch1_th.get())
				data_cont.data_list_raw[data_cont.file_index].threshold_list[1] = float(self.ch2_th.get())

				#data_list_current[file_index].datasets_list[rep].channels_list[0].fluct_arr.y = y1z
				#data_list_current[file_index].datasets_list[rep].channels_list[1].fluct_arr.y = y2z



			


			if self.normalization_index == "manual":



				y1m = y1/np.mean(y1)
				y2m = y2/np.mean(y2)

				

				data_cont.data_list_raw[data_cont.file_index].threshold_list[0] = float(self.ch1_th.get())
				data_cont.data_list_raw[data_cont.file_index].threshold_list[1] = float(self.ch2_th.get())
				

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

		if data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] != None:
			data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] = None
		
		self.Peaks()



		self.Fitting_frame()


	

	def Normalize_for_plot_index(self, event):
		
		self.normalization_index_for_plot = self.Normalization_for_plot.get()

		self.Peaks()
		

	def Choose_components (self, event):

		self.Fitting_frame()

	def Plot_trace(self, event):










		



		index = self.tree_t.selection()
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




		

		data_cont.file_index = file1-1
		data_cont.rep_index = rep1-1

	

		current_repetitions_number = data_cont.data_list_raw[data_cont.file_index].repetitions



		divisors = []
		for divdiv in range(1, current_repetitions_number+1):
			if current_repetitions_number % divdiv == 0:
				divisors.append(divdiv)

		self.Binning_choice.config(values = divisors)
		self.Binning_choice.set(data_cont.data_list_raw[data_cont.file_index].binning)


		self.channel_pairs = []
		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
				for j in range (i+1, data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
					str1 = data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[i].short_name + "/" + data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[j].short_name
					self.channel_pairs.append(str1)


			self.Channel_pair__choice.config(values = self.channel_pairs)
			self.Channel_pair__choice.set(self.channel_pairs[0])


		rep = rep1-1


		if data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index] != None:


			if len(data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index].keys()) == 3:

				

				self.Components.set("1 component")

			if len(data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index].keys()) == 6:

				

				self.Components.set("2 components")

			if len(data_cont.data_list_raw[data_cont.file_index].gp_fitting[data_cont.rep_index].keys()) == 9:

				self.Components.set("3 components")

		#self.Normalize()

		self.Peaks()

		self.Fitting_frame()



	
	def Binning(self, event):


		data_cont.change_normal = True


		data_cont.data_list_raw[data_cont.file_index].binning = int(self.Binning_choice.get())

		

		self.Peaks()

	def __init__(self, win_width, win_height, dpi_all):


		
		self.save_plot_dict = {}

		self.fit_all_flag = False
		self.normalization_index = "z-score"

		self.normalization_index_for_plot = "raw"


		self.gp_histogram = []

		self.gp_xbins = []


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

		self.hist1.set_title("Peak intensity histogram")

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

		self.frame00000001 = tk.Frame(self.frame000)
		self.frame00000001.pack(side = "top", anchor = "nw")

		self.Export_plot_button = tk.Button(self.frame00000001, text="Save plot data", command=self.Save_plot_data)
		self.Export_plot_button.pack(side = "left", anchor = "nw")

		self.plot_var = {}

		self.plot_var["Traces"] = tk.IntVar()
		self.plot_var["Peaks"] = tk.IntVar()
		self.plot_var["Intensity Histogram"] = tk.IntVar()
		self.plot_var["Dot Plot"] = tk.IntVar()
		self.plot_var["GP Plot"] = tk.IntVar()
		self.plot_var["GP Fit"] = tk.IntVar()

		for key in self.plot_var.keys():
			self.plot_var[key].set(1)

		self.traces_check=tk.Checkbutton(self.frame00000001, text="Traces", variable=self.plot_var["Traces"], command=self.Temp)
		self.traces_check.pack(side = "left", anchor = "nw")

		self.peaks_check=tk.Checkbutton(self.frame00000001, text="Peaks", variable=self.plot_var["Peaks"], command=self.Temp)
		self.peaks_check.pack(side = "left", anchor = "nw")

		self.int_hist_check=tk.Checkbutton(self.frame00000001, text="Peak intensity Histogram", variable=self.plot_var["Intensity Histogram"], command=self.Temp)
		self.int_hist_check.pack(side = "left", anchor = "nw")

		self.dot_plot_check=tk.Checkbutton(self.frame00000001, text="Dot Plot", variable=self.plot_var["Dot Plot"], command=self.Temp)
		self.dot_plot_check.pack(side = "left", anchor = "nw")

		self.gp_plot_check=tk.Checkbutton(self.frame00000001, text="GP Plot", variable=self.plot_var["GP Plot"], command=self.Temp)
		self.gp_plot_check.pack(side = "left", anchor = "nw")

		self.gp_fit_check=tk.Checkbutton(self.frame00000001, text="GP Fit", variable=self.plot_var["GP Fit"], command=self.Temp)
		self.gp_fit_check.pack(side = "left", anchor = "nw")

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


		self.Norm_label = tk.Label(self.frame001, text="Plot histogram: ")
		self.Norm_label.grid(row = 2, column = 0, sticky = 'ew')

		self.Normalization_for_plot = ttk.Combobox(self.frame001,values = ["Peak Intensity", "Peak Prominence", "Peak width at half max"], width = 9 )
		self.Normalization_for_plot.config(state = "readonly")
		
		self.Normalization_for_plot.grid(row = 2, column = 1, sticky = 'ew')

		self.Normalization_for_plot.set("Peak Intensity")

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


		for i in range(0, len(data_cont.tree_list_name)):
			name = data_cont.tree_list_name[i]
			treetree = d_tree.Data_tree (self.tree_t, name, data_cont.data_list_raw[i].repetitions)

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
		if data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number > 1:
			for i in range (data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
				for j in range (i+1, data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_number):
					str1 = data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[i].short_name + "/" + data_cont.data_list_raw[data_cont.file_index].datasets_list[0].channels_list[j].short_name
					self.channel_pairs.append(str1)

		self.tree_t.selection_set(treetree.child_id)

		



	def Temp(self):
		print("Temp function called")

		for key in self.plot_var.keys():
			print(key, self.plot_var[key].get())