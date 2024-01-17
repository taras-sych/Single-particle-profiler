import csv

# Open the original CSV file and read its contents
with open('LP_widthh.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    rows = [row for row in csvreader]

# Find the index of the label column and the starting index of the channels
label_index = rows[0].index('Compositions')
channel_start_index = 1  # assuming the first column is always 'label'

# Loop over each sample row and normalize the channel values
for row in rows[1:]:
    max_intensity = max(float(val) for val in row[channel_start_index:])
    for i in range(channel_start_index, len(row)):
        row[i] = str(float(row[i]) / max_intensity)

# Write the normalized data to a new CSV file
with open('LP_widthh_normalized.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)
