
from collections.abc import Sequence
import multiprocessing
from datetime import datetime
import pickle
import traceback

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn


from cubes import *
from sequences import MoveSequence
from solvers import *
from resources import *



import time
startTime = time.time()


## Metrics for solver evaluation ###################################################################


class SuccessRateMetric:
	"""
	Generates `samples` starting positions scrambled by `scramblingMoves` moves,
	measures how often can the solver reach solved position.
	"""

	def __init__(self,
		scramblingMoves,
		samples = 100,
		seqLength = 100,
		measureFails = False,
		measureMoves = True,
		threads = 1,
		seed = None,
	):
		self.scramblingMoves = scramblingMoves
		self.samples = samples
		self.seqLength = seqLength
		self.measureFails = measureFails
		self.measureMoves = measureMoves
		self.threads = threads
		self.seed = seed

	def __call__(self, solver, **kwargs):
		rng = np.random.default_rng(self.seed)

		if self.threads > 1:
			threadSamples = (self.samples+self.threads-1) // self.threads
			worker = _SuccessRateWorker(solver, threadSamples, self.scramblingMoves, self.seqLength)
			seeds = [rng.integers(0x7FFFFFFFFFFFFFFF) for _ in range(self.threads)]
			with multiprocessing.Pool(self.threads) as pool:
				results = pool.map(worker, seeds)
			successRate = sum(r[0] for r in results) / (self.threads*threadSamples)
			solvedInds = sum((r[1] for r in results), [])

		else:
			numSolved = 0
			solvedInds = []
			for seqInd in range(self.samples):
				seq = solver.generateSequence(
					numMoves = self.seqLength,
					init = [
						CubeTransform(CubeTransformMethod.reset, {}),
						CubeTransform(CubeTransformMethod.scramble, {"moves": self.scramblingMoves}),
					],
					seed = rng.integers(0x7FFFFFFFFFFFFFFF),
				)
				solvedInd = seq.getSolvedIndex()
				if solvedInd >= 0:
					numSolved += 1
					solvedInds.append(solvedInd)
			successRate = numSolved / self.samples

		rate = 1-successRate if self.measureFails else successRate
		if self.measureMoves:
			return rate, np.median(solvedInds) if solvedInds else np.nan
		else:
			return rate


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
		solvedInds = []
		for seqInd in range(self.numSamples):
			seq = self.solver.generateSequence(
				numMoves = self.seqLength,
				init = [
					CubeTransform(CubeTransformMethod.reset, {}),
					CubeTransform(CubeTransformMethod.scramble, {"moves": self.scramblingMoves}),
				],
				seed = rng.integers(0x7FFFFFFFFFFFFFFF),
			)
			solvedInd = seq.getSolvedIndex()
			if solvedInd >= 0:
				numSolved += 1
				solvedInds.append(solvedInd)
		return numSolved, solvedInds


def learningRateMetric(solver, **kwargs):
	if hasattr(solver, "getLearningRate"):
		return solver.getLearningRate()
	else:
		return np.nan


def numWeightsMetric(solver, **kwargs):
	if hasattr(solver, "getNumWeights"):
		return solver.getNumWeights()
	else:
		return np.nan


def memorySizeMetric(solver, **kwargs):
	if hasattr(solver, "getMemorySize"):
		return solver.getMemorySize()
	else:
		return np.nan


def trainingDataAmountMetric(seqInd, **kwargs):
	return seqInd+1

def trainingDurationMetric(trainingDuration, **kwargs):
	return trainingDuration



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
		seq = trainingSeqGenerator()
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
	metricValues = {}
	for name in metrics.keys():
		if isinstance(name, tuple):
			metricValues.update({ subName: [] for subName in name })
		else:
			metricValues[name] = []

	trainingSeqCounts = np.array(trainingSeqCounts, dtype=int)
	assert (trainingSeqCounts[1:] >= trainingSeqCounts[:-1]).all(), "Training counts must be a growing sequence"
	seqInd = 0
	evalInd = 0
	trainingDuration = 0

	progressbar = Progressbar("measure: ", samples=trainingSeqCounts[-1], evals=len(trainingSeqCounts))
	for nextSeqCnt in trainingSeqCounts:

		# training
		trainingStart = time.time()
		for seqInd in range(seqInd, nextSeqCnt):
			seq = trainingSeqGenerator()
			solver.trainOnSequence(seq)
			if seqInd % 100 == 0: # do not eval fancy progressbar too often
				progressbar(samples=seqInd+1)
		trainingDuration += time.time() - trainingStart
		progressbar(samples=seqInd+1)

		# measurement
		for name, metric in metrics.items():
			value = metric(
				solver = solver,
				seqInd = seqInd,
				evalInd = evalInd,
				trainingDuration = trainingDuration,
			)
			if isinstance(name, tuple):
				assert len(name) == len(value), \
				       f"Expected {len(name)} metric values, found {len(value)}"
				for subName, subValue in zip(name, value):
					metricValues[subName].append(subValue)
			else:
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



class MyReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):

	def __init__(self, metric, metricKwargs, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.my_metric = metric
		self.my_metric_kwargs = metricKwargs

	def step(self):
		loss = self.my_metric(**self.my_metric_kwargs)
		return super().step(loss)





# Train a new solver
#loadSolver = None

# Load a saved solver
loadSolver = "./trained_solvers/bootstrap200_906weights.pickle"

if loadSolver:

	## Load solver #################################################################################


	with open(loadSolver, "rb") as f:
		solverData = pickle.load(f)

	cube = solverData["cube"]
	solver = solverData["solver"]


else:

	## Train solver ################################################################################


	canonization = {
#		"rotationEquivalent": True,
#		"colorSwapEquivalent": 1,
	}

	print("\ncanonization:", canonization)



#	cube = Rubik_2x2x2()
	cube = Rubik_2x2x2(fixCorner=True)

	print("\ncube:", cube)

#	cube.repl()



	## Counting reachable positions ########################

#	total = 0
#	for numMoves, positions in enumerate(cube.getPositionsAfter(15, **canonization)):
#		new = len(positions)
#		total += new
#		print(f"{numMoves} moves: {new} new positions, {total} total")



	## Define a solver #####################################


#	solver = MemorizeSolver(cube, canonization, random=True)


	width = 50
	depth = 3
	colorEnc = "one-hot"
	activation = nn.ReLU

	solver = TorchMLPSolver(cube, canonization,
#		loss = ClippedLoss(torch.nn.CrossEntropyLoss(reduction="none"), 0.7, 5),
		colorEncoding = colorEnc,
	)

	net = nn.Sequential(
		nn.Linear(solver.numFeatures, width),
		activation(),
	)
	for _ in range(1, depth):
		net.append(nn.Linear(width, width))
		net.append(activation())
	net.append(nn.Linear(width, solver.numMoves))

	solver.setModel(net)

	optimizer = torch.optim.Adam(solver.getParam(), lr=1e-3)
	solver.setOptimizer(optimizer)


#	schedulerParams = {
#		"milestones": [10, 30, 100, 300, 1000, 3000],
#		"gamma": 0.5,
#	}
#	scheduler = torch.optim.lr_scheduler.MultiStepLR(
#		optimizer,
#		**schedulerParams,
#	)
#	solver.setScheduler(scheduler)

	schedulerParams = {
		"patience": 3,
		"factor": 0.5,
	}
	scheduler = MyReduceLROnPlateau(
		SuccessRateMetric(
			scramblingMoves = 10,
			samples = 500,
			measureFails = True,
			measureMoves = False,
			threads = 6,
			seed = np.random.randint(0x7FFFFFFFFFFFFFFF),
		),
		{ "solver": solver },
		optimizer,
		**schedulerParams,
	)
	solver.setScheduler(scheduler, 10000)


	print("\nsolver:", solver)
	print("scheduler params:", schedulerParams)



	## Train the solver ####################################


	class InverseScrambleGenerator:

		def __init__(self, solver, numScrambleMoves=10):
			self.randomSolver = RandomSolver(solver.cube)
			self.numScrambleMoves = numScrambleMoves

		def __call__(self):
			seq = self.randomSolver.generateSequence(
				numMoves = self.numScrambleMoves,
				init = CubeTransform(CubeTransformMethod.reset, {}),
			)
			return seq.invert().canonize(**canonization).check()

		def earlyStop(self):
			return False

		def __str__(self):
			return f"{self.__class__.__name__}(numScrambleMoves={self.numScrambleMoves})"


	class BootstrapGenerator:

		def __init__(self, solver, maxScMoves=50, testMoves=100):
			self.solver = solver
			self.maxScMoves = maxScMoves
			self.testMoves = testMoves
			self.numGenerated = 0
			self.meanScMoves = 1
			self.updateSpeed = 0.01
			self.targetSR = 0.15

		def __call__(self):
			for attempt in range(300):
#				scMoves = np.random.poisson(lam = self.meanScMoves) + 1
				scMoves = int(min(self.meanScMoves, self.maxScMoves))
				seq = self.solver.generateSequence(
					numMoves = self.testMoves,
					init = [
						CubeTransform(CubeTransformMethod.reset, {}),
						CubeTransform(CubeTransformMethod.scramble, {"moves": scMoves}),
					],
				)

				if seq.isSolved():
					self.meanScMoves += self.updateSpeed
					statSpeed = 0.001
					self.numGenerated += 1
					return seq.simplify()
				else:
					self.meanScMoves = max(self.meanScMoves-self.updateSpeed*self.targetSR, 1)

			raise RuntimeError("Could not generate a successfull sequence")

		def earlyStop(self):
			return self.meanScMoves > self.maxScMoves*10

		def __str__(self):
			return f"{self.__class__.__name__}(maxScMoves={self.maxScMoves}, testMoves={self.testMoves})"


	trainSeqGen = InverseScrambleGenerator(solver)
#	trainSeqGen = BootstrapGenerator(solver)

	seqCounts = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000,
	             1200000, 1500000, 2000000, 2500000, 3000000, 4000000, 5000000]

	print("\ndata generator:", trainSeqGen, "\n")



	metricValues = dependencyOnTrainingData(
		solver,
		seqCounts,
		trainSeqGen,
		{
			("success rate 5", "moves needed 5"):
				SuccessRateMetric( 5, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
			("success rate 10", "moves needed 10"):
				SuccessRateMetric(10, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
			("success rate 20", "moves needed 20"):
				SuccessRateMetric(20, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
			("success rate 100", "moves needed 100"):
				SuccessRateMetric(100, 500, threads=6, seed=np.random.randint(0x7FFFFFFFFFFFFFFF)),
			"learning rate": learningRateMetric,
			"memory size": memorySizeMetric,
			"num weights": numWeightsMetric,
			"train sequences": trainingDataAmountMetric,
			"training duration": trainingDurationMetric,
		},
		earlyStop = lambda metricValues, **kwargs: metricValues["learning rate"][-1] < 1e-7 or trainSeqGen.earlyStop(),
	)



	for name in ["memory size", "num weights", "learning rate", "train sequences", "training duration"]:
		print(f"\n{name}:\n\t" + "\n\t".join(map(str, metricValues[name])))

	plt.subplot(211)
	for name in ["success rate 5", "success rate 10", "success rate 20", "success rate 100"]:
		print(f"\n{name}:\n\t" + "\n\t".join(map(str, metricValues[name])))
		plt.semilogx(np.maximum(metricValues["train sequences"], 1), metricValues[name], label=name)
	plt.ylim(0, 1)
	plt.grid()
	plt.legend()

	plt.subplot(212)
	for name in ["moves needed 5", "moves needed 10", "moves needed 20", "moves needed 100"]:
		print(f"\n{name}:\n\t" + "\n\t".join(map(str, metricValues[name])))
		plt.semilogx(np.maximum(metricValues["train sequences"], 1), metricValues[name], label=name)
	plt.grid()
	plt.legend()

	plt.show()


	def finalMeas():
		for scMoves in list(range(2, 31, 2))+[50, 100]:
			metric = SuccessRateMetric(scMoves, 5000, threads=6)
			solved, moves = metric(solver)
			print(f"  success rate {scMoves} = {solved:.3f}, median moves {moves}")

	print("\nFully trained solver performance:")
	finalMeas()

	print(f"\nduration: {round((time.time()-startTime)/60)} min")

	saveNote = ""
	savePath = saveNote + datetime.now().strftime("%Y-%M-%d_%H.%m.%S") + ".pickle"
	breakpoint()
	try:
		with open(savePath, "wb") as f:
			pickle.dump({
				"solver": solver,
				"cube": cube,
				"canonization": canonization,
				"metric values": metricValues,
			}, f)
	except:
		traceback.print_exc()



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

