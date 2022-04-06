class Data_tree:
	

	def __init__(self, tree, name, repetitions):
		

		
		
		
		self.folder1=tree.insert( "", "end", text=name)
		self.child_id = tree.get_children()[-1]
		for i in range(0, repetitions):
			text1 = "repetition " + str (i+1)
			tree.insert(self.folder1, "end", text=text1)

		#tree.focus(child_id)
		#tree.selection_set(child_id)


class Data_tree_fcs_fit:

	def __init__(self, tree, name, dataset):
		

		
		
		
		self.folder1=tree.insert( "", "end", text=name)
		self.child_id = tree.get_children()[-1]
		for i in range(0, dataset.repetitions):
			text1 = "repetition " + str (i+1)
			self.folder2=tree.insert(self.folder1, "end", text=text1)

			for j in range(dataset.datasets_list[i].channels_number):
				text1 = dataset.datasets_list[i].channels_list[j].short_name
				tree.insert(self.folder2, "end", text = text1)

			for j in range(dataset.datasets_list[i].cross_number):
				text1 = dataset.datasets_list[i].cross_list[j].short_name
				tree.insert(self.folder2, "end", text = text1)