import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('LP_widthh.csv')

# Extract the labels from the 'Compositions' column as a Categorical series with ordered=True
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
fig, ax1 = plt.subplots(figsize=(20, 11.25))
fig.subplots_adjust(left=0.05, right=0.7, bottom=0.05, top=0.95)

ax1.scatter(transformed_data[:, 0], transformed_data[:, 1], c=colors, cmap='tab10', alpha=0.8)

# Add legend to the plot
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='32')

# Remove the axis labels
ax1.set_xticks([])
ax1.set_yticks([])

# Save the plot with higher dpi
plt.savefig('UMAP_LP_widthh.png', dpi=600, bbox_inches='tight', transparent = True)

plt.close()
