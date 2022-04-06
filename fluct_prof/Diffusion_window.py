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