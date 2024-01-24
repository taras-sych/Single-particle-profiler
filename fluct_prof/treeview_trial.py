import tkinter as tk
from tkinter import tix

def on_select(event):
    selected_items = tree.getcurselection()
    for item in selected_items:
        tree.tag_configure(item, background='lightblue')

# Create a Tkinter window
root = tk.Tk()
root.title("CheckboxTreeview Example")

# Create a CheckboxTreeview with an additional column for checkboxes
tree = tix.CheckList(root)
tree.pack(expand=1, fill='both')
tree.hlist["columns"] = ("Name", "Age")

# Add some sample data
tree.hlist.add("item1", text="John", values=("25",))
tree.hlist.add("item2", text="Jane", values=("30",))
tree.hlist.add("item3", text="Bob", values=("22",))

# Bind the selection event to the on_select function
tree.bind("<ButtonRelease-1>", on_select)

# Configure the column headings
tree.hlist.heading("#0", text="Checkbox")
tree.hlist.heading("Name", text="Name")
tree.hlist.heading("Age", text="Age")

# Start the Tkinter event loop
root.mainloop()
