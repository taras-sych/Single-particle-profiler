import tkinter as tk

from tkinter import ttk

from fluct_prof import Main_window as main_w

global root
global win_width
global win_height
global data_frame

def Create_root():

	global root 
	global win_width
	global win_height
	global data_frame

	root = tk.Tk()
	root.title("Single Particle Profiler")

	screen_width = root.winfo_screenwidth()
	screen_height = root.winfo_screenheight()

	win_width = round(0.5 * screen_width)
	win_height = round (0.8 * screen_height)

	#fontsize = round(win_width/85)

	#helv36 = tkFont.Font(size=fontsize)

	line = str(win_width) + "x" + str(win_height)


	root.geometry(line)

	tabs = ttk.Notebook(root, width=win_width, height=win_height, padding = 0)

	tab = []

	frame0 = tk.Frame(tabs)
	frame1 = tk.Frame(tabs)


	frame0_l = tk.LabelFrame(frame0)
	frame0_l.pack(side = "left", anchor = "nw", expand = 1, fill = tk.BOTH)
	frame0_l.config(bd=0, width = round(win_width * 0.5), height = win_height)
	frame0_l.grid_propagate(1)

	frame0_r = tk.LabelFrame(frame0)
	frame0_r.pack(side = "left", anchor = "nw", expand = 1, fill = tk.BOTH)
	frame0_r.config(bd=0, width = round(win_width * 0.5), height = win_height)
	frame0_r.grid_propagate(1)

	frame1_l = tk.LabelFrame(frame1)
	frame1_l.pack(side = "left", anchor = "nw", expand = 1, fill = tk.BOTH)
	frame1_l.config(bd=0, width = round(win_width * 0.5), height = win_height)
	frame1_l.grid_propagate(1)

	frame1_r = tk.LabelFrame(frame1)
	frame1_r.pack(side = "left", anchor = "nw", expand = 1, fill = tk.BOTH)
	frame1_r.config(bd=0, width = round(win_width * 0.5), height = win_height)
	frame1_r.grid_propagate(1)



	tabs.add(frame0, text = "SPP")
	#tabs.add(frame1, text = "Scanning FCS")

	tabs_number = 2;

	tabs.pack(side = "left", anchor = "nw")



	data_frame = main_w.Left_frame(frame0_l, win_width, win_height, dpi_all )

	data_frame_sFCS = main_w.sFCS_frame(frame1_l, win_width, win_height, dpi_all )

binning_list = []

file_index = 0
rep_index = 0

tree_list = []


tree_list_name = []

output_file_name = ''

fit_list_x = []
fit_list_y = []

Fit_params = 0


initialdirectory = ''

change_normal = False


list_of_channel_pairs = []

gp_list = []

peaks_list = []

data_list_raw = []


data_list_current = []

repetitions_list = []
total_channels_list = []

dpi_all = 75

dirdir = dir()

