import fcs_importer

import codecs

filename = "C:\\Users\\taras.sych\\Science\\Program development\\FCCS as FACS\\2021.03.25 - PyProfiler v1.2\\EVs 57 mCherry + Pro12A high.fcs"

file = codecs.open (filename, encoding='latin')

lines = file.readlines()

if filename.endswith('.fcs'):
    print(fcs_importer.Find_channels_and_repetitions (lines))