from fluct_prof import fcs_importer_dev

import tkinter as tk

import codecs

import os

import tkinter.filedialog



initialdirectory = __file__

ftypes = [('FCS .fcs', '*.fcs'), ('FCS .SIN', '*.SIN'), ('Text files', '*.txt'), ('All files', '*'), ]
		

filename =  tk.filedialog.askopenfilenames(initialdir=os.path.dirname(initialdirectory),title = "Select file", filetypes = ftypes)

file = codecs.open (filename[0], encoding='latin')

lines = file.readlines()

fcs_importer_dev.Fill_dataset_fcs(lines)