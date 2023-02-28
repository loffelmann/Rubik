
from time import time
import numpy as np



class Progressbar:
	"""
	Fancy progressbar which tries to adapt to different types of operations having different duration.
	Construct with `finalValues` containing the target value for each operation type, then update
	by calling the instance with current `values` (not all operations have to be updated each time).
	"""

	def __init__(self, progressbarMessage="", progressbarLength=100, **finalValues):
		self.message = progressbarMessage
		self.length = progressbarLength
		self.names = sorted(finalValues.keys())

		self.finalValues = np.array([finalValues[name] for name in self.names])
		self.values = np.zeros(len(self.names))
		self.weights = np.zeros(len(self.names))

		self.time0 = time()
		redundancy = 2
		self.times = np.zeros(len(self.names)*redundancy)
		self.matrix = np.zeros((len(self.names)*redundancy, len(self.names)))

		self.lastCnt = 0
		self.print()


	def __call__(self, **values):
		for i, name in enumerate(self.names):
			self.values[i] = values.get(name, self.values[i])

		# reconstruct how much time each type of operation takes (`weights`)

		reg = 1e-3 # regularization parameter (promoting closeness to previous value)
		currentT = time() - self.time0

		# assemble a "values * value weights = current time" equation system
		# ?? find a more intelligent way to construct the matrix than keeping last n data points?
		self.matrix[1:, :] = self.matrix[:-1, :]
		self.matrix[0, :] = self.values
		self.times[1:] = self.times[:-1]
		self.times[0] = currentT

		# add Tikhonov regularization, solve
		left = self.matrix.T @ self.matrix + np.diag(reg*np.ones(len(self.names)))
		right = self.matrix.T @ self.times + reg * self.weights
		self.weights = np.linalg.solve(left, right)

		# estimate total duration, use to update progress bar

		finalT = np.dot(self.finalValues, self.weights)

		cnt = round(currentT * self.length / finalT)
		cnt = min(cnt, self.length)
		if cnt > self.lastCnt:
			self.lastCnt = cnt
			self.print()
			if cnt == self.length:
				print()


	def __str__(self):
		return f"{self.message}[{'#'*self.lastCnt}{'-'*(self.length-self.lastCnt)}]"

	def print(self):
		print(f"\r{self}", end="", flush=True)


