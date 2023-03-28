
import numpy as np

import torch

from cubes import *
from sequences import MoveSequence



class RubikSolver:
	"""
	Base class for Rubik's cube solvers
	"""

	def __init__(self, cube, canonization={}, seed=None):
		self.cube = cube
		self.moves = cube.solvingMoves
		self.invMoves = { move: i for i, move in enumerate(self.moves) }
		self.rng = np.random.default_rng(seed)

		# canonization options specified here will be applied automatically before
		# each call of the solving algorithm implemented in `self._generateMoves`
		self.canonization = canonization

		self.reset()


	def reset(self):
		self._moveBuffer = []
		self._justCanonized = False


	def makeMove(self):
		"""
		Advances the linked cube by one move
		(depending on situation either canonization or solving)
		"""
		if not self._moveBuffer and self.canonization and not self._justCanonized:
			_, moves = self.cube.canonizePosition(**self.canonization)
			self._moveBuffer.extend(moves)
			self._justCanonized = True
		if not self._moveBuffer:
#			assert self.cube.isCanonicPosition(position, **self.canonization) # debug
			moves = self._generateMoves()
			if isinstance(moves, Move):
				self._moveBuffer.append(moves)
			else:
				self._moveBuffer.extend(moves)
			self._justCanonized = False
		move = self._moveBuffer.pop(0)
		self.cube.move(move)
		return move


	def generateSequence(self, *, numMoves=100, init=[], seed=None):
		"""
		Generates the specified number of moves and makes a `MoveSequence` out of them
		"""
		if seed is not None:
			self.rng = np.random.default_rng(seed)
		self.cube.init(init, seed=self.rng.integers(0x7FFFFFFFFFFFFFFF))
		self.reset()

		initPosition = self.cube.getPosition()
		seq = MoveSequence(
			cube = self.cube,
			positions = np.empty((numMoves+1, len(initPosition)), dtype=initPosition.dtype),
			moves = [],
		)
		seq.positions[0, :] = initPosition
		for i in range(1, numMoves+1):
			move = self.makeMove()
			seq.positions[i, :] = self.cube.getPosition()
			seq.moves.append(move)
		return seq


	# "abstract" methods

	def _generateMoves(self, position=None):
		"""
		Runs whatever solving algorithm the specific solver implements,
		returns one or more `Move`s to be performed.
		No need to solve the cube immediately, the method will be called over and over
		(with canonization between calls).
		"""
		raise NotImplementedError(f"{self.__class__.__name__}._generateMoves")


####################################################################################################


class RandomSolver(RubikSolver):
	"""
	Selects a random solving move at each step
	"""

	def _generateMoves(self, position=None):
		moveInd = self.rng.integers(len(self.moves))
		return self.moves[moveInd]


####################################################################################################


class MemorizeSolver(RubikSolver):
	"""
	Remembers which move follows each cube position in training sequences.
	Returns remembered move for known positions, noMove or random move for unknown ones.
	"""

	def __init__(self, *args, random=False, **kwargs):
		super().__init__(*args, **kwargs)
		self.random = random
		self.memory = {}
		self.found = self.notFound = 0

	def trainOnSequence(self, seq, **whatever):
		"""
		Remembers moves following positions in `seq`
		"""
		for pos, move in zip(seq.positions, seq.moves):
			if move in self.moves:
				self.memory[tuple(pos)] = move

	def _generateMoves(self, position=None):
		if position is None:
			position = self.cube.position
		if self.cube.isSolved(position=position):
			return noMove
		position = tuple(position)
		if position in self.memory:
			self.found += 1
			return self.memory[tuple(position)]
		else:
			self.notFound += 1
			if self.random:
				return self.moves[self.rng.integers(len(self.moves))] # random instead of noMove
			else:
				return noMove

	def getMemorySize(self):
		return len(self.memory)


####################################################################################################


class _TorchWrapper(torch.nn.Module):

	def __init__(self, net):
		super().__init__()
		self.net = net

	def forward(self, x):
		return self.net(x)


class TorchMLPSolver(RubikSolver):

	def __init__(self,
		cube,
		canonization = {},
		*,
		device = "cpu",
		loss = None,
		predictMode = "probability",
		colorEncoding = None,
	):
		super().__init__(cube, canonization)
		self.predictMode = predictMode

		self.loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
		self.temperature = 1.0

		self.model = None
		self.device = device
		self.optimizer = None
		self.scheduler = None
		self.epochLen = 1000

		if colorEncoding is not None:
			self.colorEncoding = { c: torch.tensor(e)
			                       for c, e in colorEncoding.items() }
		else: # generate one-hot encoding
			self.colorEncoding = {}
			for i, c in enumerate(self.cube.faceValues):
				self.colorEncoding[c] = torch.zeros(len(self.cube.faceValues))
				self.colorEncoding[c][i] = 1.0

		self.numFeatures = len(self.getFeatures())
		self.numMoves = len(self.moves)

		self.numSteps = 0

#		self.lrSpeed = 0.1


	def setModel(self, model, device=None):
		if device is not None:
			self.device = device
		self.model = _TorchWrapper(model).to(self.device)
		self.model.train()
		self._training = True

	def getParam(self):
		if self.model is None:
			raise RuntimeError("Cannot get parameters without setting the NN first")
		return self.model.parameters()

	def setOptimizer(self, optimizer):
		self.optimizer = optimizer

	def setScheduler(self, scheduler, epochLen=None):
		self.scheduler = scheduler
		if epochLen is not None:
			self.epochLen = epochLen


	def __str__(self):
		return f"""{self.__class__.__name__}(
  model={self.model.net}
  loss={self.loss}
  optimizer={self.optimizer}
  scheduler={self.scheduler}
  epochLen={self.epochLen}
)"""


	def getFeatures(self, position=None):
		"""
		Generates NN input for a single cube position
		"""
		if position is None:
			position = self.cube.position
		return torch.cat([self.colorEncoding[c] for c in position])


	def sequence2TrainData(self, seq):
		"""
		Filters out dummy and canonization moves (whole-cube rotations, color sorting);
		Converts remaining positions to numpy features and moves to numpy targets.
		"""
		numSolvingMoves = sum((m in self.moves) for m in seq.moves)
		features = torch.empty((numSolvingMoves, len(self.getFeatures())))
		targets = torch.zeros((numSolvingMoves, len(self.moves)))
		moveInd = 0
		for i, move in enumerate(seq.moves):
			if move not in self.moves:
				continue
			features[moveInd, :] = self.getFeatures(seq.positions[i, :])
			targets[moveInd, self.invMoves[move]] = 1
			moveInd += 1

		return features.to(self.device), targets.to(self.device)


#	def raiseLearningRate(self):
#		for group in self.optimizer.param_groups:
#			group["lr"] = min(group["lr"]*(1+self.lrSpeed), 1e-2)
#
#	def lowerLearningRate(self):
#		for group in self.optimizer.param_groups:
#			group["lr"] = max(group["lr"]/(1+self.lrSpeed), 1e-7)

	def getLearningRate(self):
		minLR = np.inf
		maxLR = -np.inf
		for group in self.optimizer.param_groups:
			minLR = min(minLR, group["lr"])
			maxLR = max(maxLR, group["lr"])
		if minLR == maxLR:
			return minLR
		return (minLR, maxLR)

	def getNumWeights(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


	def trainOnSequence(self, seq):
		"""
		Converts `seq` to (position, next move) pairs, runs one iteration of optimizer on it
		"""
		self.numSteps += 1

		if not self._training:
			self.model.train()
			self._training = True
		features, targets = self.sequence2TrainData(seq)

		prediction = self.model(features)
		loss = self.loss(prediction, targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if self.scheduler is not None and self.numSteps % self.epochLen == 0:
			self.scheduler.step()


	def _generateMoves(self, position=None):
		if position is None:
			position = self.cube.position

		if self.cube.isSolved(position=position):
			return noMove

		if self._training:
			self.model.eval()
			self._training = False
		output = self.model(self.getFeatures()[None, :])

		if self.predictMode == "maximum":
			index = torch.argmax(output).item()
			return self.moves[index]
		elif self.predictMode == "probability":
			probability = softmax(output[0].detach().numpy() / self.temperature)
			try:
				index = self.rng.choice(np.arange(len(self.moves)), p=probability)
			except:
				breakpoint()
			return self.moves[index]
		else:
			raise ValueError(self.predictMode)


def softmax(x):
	xPos = np.exp(x - x.max())
	return xPos / xPos.sum()

def softermax(x):
	xPos = (np.sqrt(x*x+1) + x) * 0.5
#	xPos = xPos - xPos.min() + 1.0
	return xPos / xPos.sum()


