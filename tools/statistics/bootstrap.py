
import numpy as np
from matplotlib import pyplot as plt

class BootStrap ():

	def __init__ (self, n_boots=None):
		self.n_boots = n_boots
		self._N = -1

	def set_nr_resamples (self, nr_resamples):
		self.n_boots = None


	def run_bootstrap_std (self, y, do_plot = False):

		self._N = len(y)
		if self.n_boots is None:
			self.n_boots = self._N

		if (self._N > 0):
			self.bootstrap_y = np.zeros((self.n_boots, self._N))
			self.yB = np.zeros(self.n_boots)
			self.std_B = np.zeros(self.n_boots)

			for alpha in np.arange(self.n_boots):
				ind = np.random.randint (0, self._N, self._N)
				self.bootstrap_y [alpha, :] = self.y[ind]
				self.yB [alpha] = np.mean (self.y[ind])
				self.std_B [alpha] = np.std(self.y[ind])

			if do_plot:
				plt.plot (self.std_B)
				plt.ylabel ('holevo var bootstrap')
				plt.show()

			self.mean_bootstrap = np.mean (self.yB)
			self.std_bootstrap = np.mean(self.std_B)
			self.err_std_bootstrap = np.std (self.std_B)

			return self.mean_bootstrap, self.std_bootstrap, self.err_std_bootstrap
		else:
			print ('Unspecified input data!')

	def run_bootstrap_CI (self, y, percentile=0.95, do_plot = False):

		self._N = len(y)
		if self.n_boots is None:
			self.n_boots = self._N

		if (self._N > 0):
			self.bootstrap_y = np.zeros((self.n_boots, self._N))
			self.yB = np.zeros(self.n_boots)
			self.delta_B = np.zeros(self.n_boots)
			y_mean = np.mean(y)

			for alpha in np.arange(self.n_boots):
				ind = np.random.randint (0, self._N, self._N)
				self.bootstrap_y [alpha, :] = y[ind]
				self.yB [alpha] = np.mean (y[ind])
				self.delta_B [alpha] = self.yB [alpha] - y_mean

			if do_plot:
				plt.plot (self.delta_B)
				plt.ylabel ('holevo var bootstrap')
				plt.show()

			# this computes the confidence interval on the mean
			D = np.sort(self.delta_B)
			Dmin = D[int(len(D)*(0.5-percentile/2.))]
			Dmax = D[int(len(D)*(0.5+percentile/2.))]
			#plt.figure (figsize=(12,4))
			#plt.hist (D, 100)
			#plt.title ("D: "+str(Dmin)+' -- '+str(Dmax))
			#plt.show()
			M = np.mean (self.yB)
			return M, np.std (self.yB), [M+Dmin, M+Dmax]
		else:
			print ('Unspecified input data!')


	def run_bootstrap_holevo (self):

		if (self._N > 0):
			self.bootstrap_y = np.zeros((self.n_boots, self._N)) + 1j*np.zeros((self.n_boots, self._N))
			self.yB = np.zeros(self.n_boots)
			self.hB = np.zeros(self.n_boots)

			for alpha in np.arange(self.n_boots):
				ind = np.random.randint (0, self._N, self._N)
				self.bootstrap_y [alpha, :] = self.y[ind]
				self.hB [alpha] = np.abs(np.mean(self.y[ind]))**(-2)-1

			self.meanH_bootstrap = np.mean(self.hB)
			self.errH_bootstrap = np.std (self.hB)

		else:
			print ('Unspecified input data!')

	def print_results (self):
		print ('Bootstrap over '+str(self.n_boots)+' resamples')
		print ('Mean: '+str(self.mean_bootstrap)+ '  StDev: '+str(self.std_bootstrap))
		print ('Error on StDev: '+str(self.err_std_bootstrap))

	def histogram (self):
		plt.figure()
		plt.hist(self.std_B)
		plt.xlabel ('bootstrap std')
		plt.show()





