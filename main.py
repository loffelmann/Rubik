
from collections.abc import Sequence
import multiprocessing
import time

import numpy as np
import matplotlib.pyplot as plt

from torch import nn


from cubes import *
from sequences import MoveSequence
from solvers import *



## Miscellaneous ###################################################################################


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

		self.time0 = time.time()
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
		currentT = time.time() - self.time0

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



## Metrics for solver evaluation ###################################################################


class SuccessRateMetric:
	"""
	Generates `samples` starting positions scrambled by `scramblingMoves` moves,
	measures how often can the solver reach solved position.
	`scramblingMoves` can be a list (a list of sucess rates is returned then).
	"""

	def __init__(self, scramblingMoves, samples=100, seqLength=100, threads=1, seed=None):
		self._multipleCounts = isinstance(scramblingMoves, Sequence)
		self.scramblingMoves = scramblingMoves if self._multipleCounts else [scramblingMoves]
		self.samples = samples
		self.seqLength = seqLength
		self.threads = threads
		self.seed = seed

	def __call__(self, solver):
		rng = np.random.default_rng(self.seed)
		rates = []
		for scMoveCount in self.scramblingMoves:

			if self.threads > 1:
				threadSamples = (self.samples+self.threads-1) // self.threads
				worker = _SuccessRateWorker(solver, threadSamples, scMoveCount, self.seqLength)
				seeds = [rng.integers(0x7FFFFFFFFFFFFFFF) for _ in range(self.threads)]
				with multiprocessing.Pool(self.threads) as pool:
					numsSolved = pool.map(worker, seeds)
				successRate = sum(numsSolved) / (self.threads*threadSamples)

			else:
				numSolved = 0
				for seqInd in range(self.samples):
					seq = solver.generateSequence(
						numMoves = self.seqLength,
						init = [
							CubeTransform(CubeTransformMethod.reset, {}),
							CubeTransform(CubeTransformMethod.scramble, {"moves": scMoveCount}),
						],
						seed = rng.integers(0x7FFFFFFFFFFFFFFF),
					)
					if seq.isSolved():
						numSolved += 1
				successRate = numSolved / self.samples

			rates.append(successRate)

		return rates if isinstance(self.scramblingMoves, Sequence) else rates[0]


class _SuccessRateWorker:
	"""
	A multiprocessing worker to parallelize `SuccessRateMetric`.
	Mostly good for solvers which are not parallel themselves, and are picklable.
	"""

	def __init__(self, solver, numSamples, scramblingMoves, seqLength):
		self.solver = solver
		self.numSamples = numSamples
		self.scramblingMoves = scramblingMoves
		self.seqLength = seqLength

	def __call__(self, seed=None):
		rng = np.random.default_rng(seed)
		numSolved = 0
		for seqInd in range(self.numSamples):
			seq = self.solver.generateSequence(
				numMoves = self.seqLength,
				init = [
					CubeTransform(CubeTransformMethod.reset, {}),
					CubeTransform(CubeTransformMethod.scramble, {"moves": self.scramblingMoves}),
				],
				seed = rng.integers(0x7FFFFFFFFFFFFFFF),
			)
			if seq.isSolved():
				numSolved += 1
		return numSolved



## Training methods ################################################################################


def train(
	solver,
	trainingSeqCount,
	trainingSeqGenerator,
):
	"""
	Feeds sequences generated by `trainingSetGenerator`
	into the `trainOnSequence` method of `solver`
	"""
	progressbar = Progressbar("train: ", samples=trainingSeqCount)
	for seqInd in range(trainingSeqCount):
		progressbar(samples=seqInd+1)
		seq = trainingSeqGenerator(solver)
		solver.trainOnSequence(seq)


def dependencyOnTrainingData(
	solver,
	trainingSeqCounts, # growing sequence expected (describes cumulative training)
	trainingSeqGenerator, # function(solver) -> MoveSequence
	metrics, # dict of "metric name": function(solver) -> value
):
	"""
	The same training routine as `train`, interrupted by metric evaluations
	"""
	metricValues = { name: [] for name in metrics.keys() }
	seqInd = 0
	evalInd = 0

	progressbar = Progressbar("measure: ", samples=trainingSeqCounts[-1], evals=len(trainingSeqCounts))
	for nextSeqCnt in trainingSeqCounts:

		# training
		for seqInd in range(seqInd, nextSeqCnt):
			seq = trainingSeqGenerator(solver)
			solver.trainOnSequence(seq)
			if seqInd % 100 == 0: # do not eval fancy progressbar too often
				progressbar(samples=seqInd+1)
		progressbar(samples=seqInd+1)

		# measurement
		for name, metric in metrics.items():
			value = metric(solver)
			metricValues[name].append(value)
		evalInd += 1
		progressbar(evals=evalInd)

	return metricValues



## Experiments #####################################################################################


randomSeed = 1
print(f"\nrandom seed = {randomSeed}")
np.random.seed(randomSeed)
# ?? does torch need seeding?


canonization = {
	"rotationEquivalent": True,
	"colorSwapEquivalent": 1,
}

print("\ncanonization:", canonization)



cube = Rubik_2x2x2()
#cube = Rubik_2x2x2(fixCorner=True)

print("\ncube:", cube)

#cube.repl()



## Counting reachable positions ############################

#total = 0
#for numMoves, positions in enumerate(cube.getPositionsAfter(15, **canonization)):
#	new = len(positions)
#	total += new
#	print(f"{numMoves} moves: {new} new positions, {total} total")



## Define a solver #########################################


#solver = MemorizeSolver(cube, canonization)


solver = TorchMLPSolver(cube, canonization)

solver.setModel(nn.Sequential(
	nn.Linear(solver.numFeatures, 50),
	nn.ReLU(),
	nn.Linear(50, 50),
	nn.ReLU(),
	nn.Linear(50, solver.numMoves),
))

optimizer = torch.optim.Adam(solver.getParam(), lr=1e-3)
solver.setOptimizer(optimizer)

schedulerParams = {
	"milestones": [10, 30, 100, 300, 1000, 3000],
	"gamma": 0.5,
}
scheduler = torch.optim.lr_scheduler.MultiStepLR(
	optimizer,
	**schedulerParams,
)
solver.setScheduler(scheduler)

print("\nsolver:", solver)
print("scheduler params: ", schedulerParams)



## Train the solver ########################################

class InverseScrambleGenerator:

	def __init__(self, cube, numScrambleMoves=10):
		self.solver = RandomSolver(cube)
		self.numScrambleMoves = numScrambleMoves

	def __call__(self, solver):
		assert isinstance(solver.cube, self.solver.cube.__class__)
		seq = self.solver.generateSequence(
			numMoves = self.numScrambleMoves,
			init = CubeTransform(CubeTransformMethod.reset, {}),
		)
		return seq.invert().canonize(**canonization).check()

	def __str__(self):
		return f"{self.__class__.__name__}(numScrambleMoves={self.numScrambleMoves})"


trainSeqGen = InverseScrambleGenerator(cube)
seqCounts = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

print("\ndata generator:", trainSeqGen, "\n")

metricValues = dependencyOnTrainingData(
	solver,
	seqCounts,
	trainSeqGen,
	{
		"success rate 5":  SuccessRateMetric( 5, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
		"success rate 10": SuccessRateMetric(10, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
		"success rate 20": SuccessRateMetric(20, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
	},
)

for name, values in metricValues.items():
	print(f"\n{name}:\n\t" + "\n\t".join(map(str, values)))
	plt.semilogx(np.maximum(seqCounts, 1), values, label=name)
print()
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.show()

breakpoint()



## Visualize solver runs ###################################

while True:
	seq = solver.generateSequence(numMoves=100, init=[
		CubeTransform(CubeTransformMethod.reset, {}),
		CubeTransform(CubeTransformMethod.scramble, {"moves": 9}),
	])
	seq = seq.simplify()
	seq.check()
	seq.animate()
	plt.show() # keep matplotlib window open

