import numpy as np
import lmfit


def Corr_curve_3d(tc, offset, GN0, A1, txy1, alpha1, AR1, B1, tauT1 ):

	txy1 = txy1 / 1000

	tauT1 = tauT1 / 1000

	G_Diff =  (A1*(((1+((tc/txy1)**alpha1))**-1)*(((1+(tc/((AR1**2)*txy1)))**-0.5))))

	G_T = 1 + (B1*np.exp(tc/(-tauT1)))

	return offset + GN0 * G_Diff * G_T

def Corr_curve_2d(tc, offset, GN0, A1, txy1, alpha1, B1, tauT1):

	txy1 = txy1 / 1000

	tauT1 = tauT1 / 1000

	G_Diff =  A1*(((1+((tc/txy1)**alpha1))**-1))

	G_T = 1 + (B1*np.exp(tc/(-tauT1)))

	return offset + GN0 * G_Diff * G_T


def Fit_Curve():


	params = lmfit.Parameters()

	row_index = 1
	for param in list_of_params:

		params.add(param, 
			float(self.full_dict[param]["Init"].get()), 
			vary = self.fixed_list[row_index-1].get(), 
			min = float(self.full_dict[param]["Min"].get()), 
			max = float(self.full_dict[param]["Max"].get()))

		row_index+=1


def resid (self, params, x, ydata ):

	param_list = []

	for param in params.keys():

		param_list.append( np.float64(params[param].value))


	

	
	
	if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component' and self.Dimension.get() == "3D":

		y_model = Corr_curve_3d(x, *param_list)

	if self.Triplet.get() == 'triplet' and self.Components.get() == '1 component' and self.Dimension.get() == "2D":
		y_model = Corr_curve_2d(x, *param_list)


	return y_model - ydata