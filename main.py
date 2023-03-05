
from collections.abc import Sequence
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from torch import nn


from cubes import *
from sequences import MoveSequence
from solvers import *
from resources import *



## Metrics for solver evaluation ###################################################################


class SuccessRateMetric:
	"""
	Generates `samples` starting positions scrambled by `scramblingMoves` moves,
	measures how often can the solver reach solved position.
	`scramblingMoves` can be a list (a list of sucess rates is returned then).
	"""

	def __init__(self, scramblingMoves, samples=100, seqLength=100, measureFails=False, threads=1, seed=None):
		self._multipleCounts = isinstance(scramblingMoves, Sequence)
		self.scramblingMoves = scramblingMoves if self._multipleCounts else [scramblingMoves]
		self.samples = samples
		self.seqLength = seqLength
		self.measureFails = measureFails
		self.threads = threads
		self.seed = seed

	def __call__(self, solver, **kwargs):
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

			rates.append(1-successRate if self.measureFails else successRate)

		return rates if self._multipleCounts else rates[0]


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


def learningRateMetric(solver, **kwargs):
	if hasattr(solver, "getLearningRate"):
		return solver.getLearningRate()
	else:
		return np.nan


def trainingDataAmountMetric(seqInd, **kwargs):
	return seqInd



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
	earlyStop = None,
):
	"""
	The same training routine as `train`, interrupted by metric evaluations
	"""
	metricValues = { name: [] for name in metrics.keys() }

	trainingSeqCounts = np.array(trainingSeqCounts, dtype=int)
	assert (trainingSeqCounts[1:] >= trainingSeqCounts[:-1]).all(), "Training counts must be a growing sequence"
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
			value = metric(
				solver = solver,
				seqInd = seqInd,
				evalInd = evalInd,
			)
			metricValues[name].append(value)
		evalInd += 1
		progressbar(evals=evalInd)

		# stopping condition
		if earlyStop is not None and earlyStop(
			seqInd = seqInd,
			evalInd = evalInd,
			metricValues = metricValues,
		):
			progressbar.end()
			break

	return metricValues



## Experiments #####################################################################################


canonization = {
#	"rotationEquivalent": True,
#	"colorSwapEquivalent": 1,
}

print("\ncanonization:", canonization)



#cube = Rubik_2x2x2()
cube = Rubik_2x2x2(fixCorner=True)

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


#schedulerParams = {
#	"milestones": [10, 30, 100, 300, 1000, 3000],
#	"gamma": 0.5,
#}
#scheduler = torch.optim.lr_scheduler.MultiStepLR(
#	optimizer,
#	**schedulerParams,
#)
#solver.setScheduler(scheduler)


class MyReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):

	def __init__(self, metric, metricKwargs, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.my_metric = metric
		self.my_metric_kwargs = metricKwargs

	def step(self):
		loss = self.my_metric(**self.my_metric_kwargs)
		return super().step(loss)

schedulerParams = {
	"patience": 3,
	"factor": 0.5,
}
scheduler = MyReduceLROnPlateau(
	SuccessRateMetric(10, 500, measureFails=True, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
	{ "solver": solver },
	optimizer,
	**schedulerParams,
)
solver.setScheduler(scheduler, 10000)


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
seqCounts = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000,
             1200000, 1500000, 2000000, 2500000, 3000000, 4000000, 5000000]

print("\ndata generator:", trainSeqGen, "\n")

metricValues = dependencyOnTrainingData(
	solver,
	seqCounts,
	trainSeqGen,
	{
		"success rate 5":  SuccessRateMetric( 5, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
		"success rate 10": SuccessRateMetric(10, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
		"success rate 20": SuccessRateMetric(20, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
		"learning rate": learningRateMetric,
		"train sequences": trainingDataAmountMetric,
	},
	earlyStop = lambda metricValues, **kwargs: metricValues["learning rate"][-1] < 1e-7,
)

name = "train sequences"
print(f"\n{name}:\n\t" + "\n\t".join(map(str, metricValues[name])))

plt.subplot(211)
for name in ["success rate 5", "success rate 10", "success rate 20"]:
	print(f"\n{name}:\n\t" + "\n\t".join(map(str, metricValues[name])))
	plt.semilogx(np.maximum(metricValues["train sequences"], 1), metricValues[name], label=name)
plt.ylim(0, 1)
plt.grid()
plt.legend()

plt.subplot(212)
name = "learning rate"
print(f"\n{name}:\n\t" + "\n\t".join(map(str, metricValues[name])))
plt.semilogx(np.maximum(metricValues["train sequences"], 1), metricValues[name], label=name)
plt.grid()
plt.legend()

plt.show()

print("\nFully trained solver performance:")
for scMoves in range(2, 31, 2):
	metric = SuccessRateMetric(scMoves, 5000, threads=6)
	value = metric(solver)
	print(f"  success rate {scMoves} = {value}")

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

