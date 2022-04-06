#Multiple files


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

from fluct_prof import Main_window as main_w

#--------------------------
#End of importing own modules
#--------------------------




sns.set(context='notebook', style='whitegrid')




root = tk.Tk()
root.title("FCS all inclusive")


screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

win_width = round(0.5 * screen_width)
win_height = round (0.8 * screen_height)

fontsize = round(win_width/85)

helv36 = tkFont.Font(size=fontsize)

line = str(win_width) + "x" + str(win_height)


root.geometry(line)

dpi_all = 75

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



tabs.add(frame0, text = "Point FCS")
tabs.add(frame1, text = "Scanning FCS")

tabs_number = 2;

tabs.pack(side = "left", anchor = "nw")



data_frame = main_w.Left_frame(frame0_l, win_width, win_height, dpi_all )


root.mainloop()