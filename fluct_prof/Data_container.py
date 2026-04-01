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
    win_height = round(0.8 * screen_height)

    root.geometry(f"{win_width}x{win_height}")
    root.minsize(1000, 700)

    tabs = ttk.Notebook(root, padding=0)
    tabs.pack(fill="both", expand=True)

    frame0 = tk.Frame(tabs)
    frame1 = tk.Frame(tabs)
    frame2 = tk.Frame(tabs)

    tabs.add(frame0, text="Single Particle Profiler")
    tabs.add(frame1, text="Scanning FCS (cross)")
    tabs.add(frame2, text="Scanning FCS (carpet)")

    data_frame = main_w.Left_frame(frame0, win_width, win_height, dpi_all)
    data_frame_sFCS = main_w.sFCS_frame(frame1, win_width, win_height, dpi_all)
    data_frame_sFCS = main_w.sFCS_carpet(frame2, win_width, win_height, dpi_all)

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

