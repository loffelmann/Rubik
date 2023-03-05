
import numpy as np
import matplotlib.pyplot as plt


from cubes import MoveType



# auxiliary for plotting

def _updatePlot():
	plt.draw()
	plt.pause(0.01)

def _getAx(ax=None):
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")
	return ax



class MoveSequence:
	"""
	Represents a sequence of Rubik's cube positions and moves between them
	"""


	def __init__(self, cube, positions=None, moves=None, scores=None):
		if positions is None:
			positions = np.empty((0, len(cube.faces)))
		if moves is None:
			moves = []
		if scores is None:
			scores = [None]*len(positions)
		self.cube = cube
		self.positions = positions.copy()
		self.moves = moves.copy()
		self.scores = scores.copy()



	@staticmethod
	def fromMoves(moves, cube, init=[]):
		"""
		Builds a `MoveSequence` from `cube`, an initial state, and a sequence of moves
		"""
		cube.rememberPosition()
		cube.init(init)
		positions = np.empty((len(moves)+1, len(cube.position)), dtype=cube.position.dtype)
		positions[0, :] = cube.position
		for i, move in enumerate(moves, start=1):
			cube.move(move)
			positions[i, :] = cube.position
		cube.restorePosition()
		return MoveSequence(
			cube = cube,
			positions = positions,
			moves = moves,
		)


	def check(self):
		"""
		Checks that positions correspond to moves
		"""
		assert len(self.positions) == len(self.moves)+1 == len(self.scores)
		self.cube.setPosition(self.positions[0, :])
		for i, move in enumerate(self.moves, start=1):
			self.cube.move(move)
			position = self.cube.getPosition()
			assert (position == self.positions[i, :]).all()
		return self


	def isSolved(self, anywhere=False):
		"""
		Checks if the seqence ends with a solved position,
		or contains a solved position when `anywhere`
		"""

		if anywhere:
			for i in range(len(self.positions)):
				if self.cube.isSolved(position=self.positions[i, :]):
					return True
			return False

		else: # checking last position only
			return self.cube.isSolved(position=self.positions[-1, :])


	def getSolvedIndex(self):
		"""
		Finds index of the first position in sequence which is solved.
		Returns -1 if no solved position found.
		"""
		for i in range(len(self.positions)):
			if self.cube.isSolved(position=self.positions[i, :]):
				return i
		return -1



	## Sequence transforms #########################################################################


	def invert(self):
		"""
		Inverts the order of positions and moves in the sequence.
		Returns a new MoveSequence, self is unchanged.
		"""
		return MoveSequence(
			cube = self.cube,
			positions = self.positions[::-1, ...],
			moves = [self.cube.invertMove(m) for m in self.moves[::-1]],
			scores = self.scores[::-1],
		)


	def canonize(self, *, keepStart=False, keepEnd=False, rotationEquivalent=False, colorSwapEquivalent=0):
		"""
		Inserts new non-solving moves (whole-cube rotations, color swaps) so that each move of the
		original sequence begins at a "canonical" cube position (lexicographically smallest from equivalent
		positions under non-solving moves).
		Returns a new MoveSequence, self is unchanged.
		"""
		canonization = { "rotationEquivalent": rotationEquivalent, "colorSwapEquivalent": colorSwapEquivalent }
		positions = []
		moves = []
		scores = []
		currentRotation = np.arange(len(self.cube.faces))
		currentColorSwap = self.cube.noColorSwap

		# canonizing the initial position
		canonizedPosition, canonizingMoves = self.cube.canonizePosition(self.positions[0], **canonization)
		if keepStart:
			raise NotImplementedError("Sequence canonization with initial position preserved")
		positions.append(canonizedPosition)
		scores.append(self.scores[0])
		for cMove in canonizingMoves:
			if cMove.type == MoveType.cubeRot:
				currentRotation = currentRotation[self.cube.moveOrders[cMove]]
			elif cMove.type == MoveType.colorSwap:
				currentColorSwap = tuple(cMove.param[c] for c in currentColorSwap)
			else:
				raise ValueError(f"Unknown canonizing move type: {cMove.type}")

		# canonizing moves and subsequent positions
		for i, move in enumerate(self.moves, start=1):
			moves.append(self.cube.rotateMove(move, currentRotation))
			position = self.cube.applyColorSwap(currentColorSwap, self.cube.applyRotation(currentRotation, self.positions[i]))
			positions.append(position)
			scores.append(self.scores[i])
			canonizedPosition, canonizingMoves = self.cube.canonizePosition(position, **canonization)
			for cMove in canonizingMoves:
				moves.append(cMove)
				if cMove.type == MoveType.cubeRot:
					position = self.cube.applyRotation(self.cube.moveOrders[cMove], position)
					currentRotation = currentRotation[self.cube.moveOrders[cMove]]
				elif cMove.type == MoveType.colorSwap:
					position = self.cube.applyColorSwap(cMove.param, position)
					currentColorSwap = tuple(cMove.param[c] for c in currentColorSwap)
				else:
					raise ValueError(f"Unknown canonizing move type: {cMove.type}")
				positions.append(position)
				scores.append(self.scores[i]) # score should not be affected by a non-solving move
			assert (position == canonizedPosition).all(), "Canonizing moves do not produce the canonized position"

		if keepEnd:
			raise NotImplementedError("Sequence canonization with final position preserved")

		return MoveSequence(
			cube = self.cube,
			positions = np.asarray(positions),
			moves = moves,
			scores = scores,
		)


	def simplify(self, *, rotations=True, colorSwaps=True):
		"""
		Removes non-solving moves like whole-cube rotations and color swaps.
		Returns a new MoveSequence, self is unchanged.
		"""
		self.cube.rememberPosition(self.positions[0])

		currentInvRotation = np.arange(len(self.cube.faces))
		currentColorSwap = self.cube.noColorSwap # only needed for check

		positions = [self.cube.getPosition()]
		moves = []
		scores = [self.scores[0]]

		for i, move in enumerate(self.moves, start=1):
			if (
				move.type == MoveType.sliceRot # always preserve
				or (not rotations and move.type == MoveType.cubeRot)
				or (not colorSwaps and move.type == MoveType.colorSwap)
			):
				move = self.cube.rotateMove(move, currentInvRotation)
				self.cube.move(move)
				moves.append(move)
				positions.append(self.cube.getPosition())

				scores.append(self.scores[i])

			elif move.type == MoveType.cubeRot:
				currentInvRotation = self.cube.moveOrders[self.cube.invertMove(move)][currentInvRotation]

			elif move.type == MoveType.colorSwap:
				currentColorSwap = tuple(move.param[c] for c in currentColorSwap) # only needed for check

			elif move.type == MoveType.none:
				pass

			else:
				raise ValueError(f"Unknown move type: {move.type}")

			# check
			origRotPos = self.cube.applyRotation(currentInvRotation, self.positions[i])
			newSwapPos = self.cube.applyColorSwap(currentColorSwap, self.cube.getPosition())
			assert (origRotPos == newSwapPos).all()

		self.cube.restorePosition()

		return MoveSequence(
			cube = self.cube,
			positions = np.asarray(positions),
			moves = moves,
			scores = scores,
		)


	def strip(self, *, start=True, end=True, everywhere=False):
		"""
		Removes moves of type "none" from sequence start, end, or everywhere in the sequence.
		Returns a new MoveSequence, self is unchanged.
		"""
		mask = np.ones(len(self.positions), dtype=bool)
		if everywhere:
			for i, move in enumerate(self.moves, start=1):
				if move.type == MoveType.none:
					mask[i] = False
		else:
			if start:
				for i, move in enumerate(self.moves, start=1):
					if move.type == MoveType.none:
						mask[i] = False
					else:
						break
			if end:
				for i, move in enumerate(self.moves[::-1], start=1):
					if move.type == MoveType.none:
						mask[-i] = False
					else:
						break
		return MoveSequence(
			cube = self.cube,
			positions = self.positions[mask, :],
			moves = [m for m, f in zip(self.moves, mask[1:]) if f],
			scores = [m for m, f in zip(self.scores, mask) if f],
		)



	## Plotting ####################################################################################


	def animate(self, ax=None):
		"""
		Displays the sequence of cube positions with animated transitions between them
		"""
		ax = _getAx(ax)
		self.cube.setPosition(self.positions[0, :])
		self.cube.plot(ax)
		_updatePlot()
		for i, move in enumerate(self.moves, start=1):
			plt.pause(0.5)
			plt.suptitle(f"{i}/{len(self.moves)}: {move}")
			self.cube.animateMove(move, ax)
			self.cube.setPosition(self.positions[i, :])
			self.cube.plot(ax)
			_updatePlot()

