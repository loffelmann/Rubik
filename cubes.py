
import itertools
from collections import namedtuple
from enum import Enum
import re

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from cy.color_eq import uniqueColors
from cy.color_eq import applyColorSwap as cy_applyColorSwap



# auxiliary for plotting

def _updatePlot():
	plt.draw()
	plt.pause(0.01)

def _getAx(ax=None):
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")
	return ax



# a type for atomic transforms of a Rubik's cube, like rotating a face by 90 degrees;
# used for solving/scrambling cubes and for moving between symmetric positions
MoveType = Enum("MoveType", ["none", "sliceRot", "cubeRot", "colorSwap"])
Move = namedtuple("Move", ["type", "param"])

noMove = Move(MoveType.none, None)

# a type for higher-level transforms of a cube, like resetting to a solved state;
# used to initialize cube state before further processing
CubeTransformMethod = Enum("CubeTransformMethod", ["reset", "scramble", "distinct"])
CubeTransform = namedtuple("CubeTransform", ["method", "kwargs"])



class RubikCuboid:

	def __init__(self,
		size,
		*,
		squareRotations = (90, 270),
		stickerFilter = None,
		maxThickness = np.inf,
		fixCorner = False,
	):
		self.fixCorner = fixCorner
		self.maxThickness = maxThickness
		self.squareRotations = tuple(((ang%360)+360)%360 for ang in squareRotations)

		if isinstance(size, int):
			size = (size, size, size)
		assert len(size) == 3
		assert isinstance(size[0], int) and isinstance(size[1], int) and isinstance(size[2], int)
		self.size = tuple(size)

		self.stickerFilter = stickerFilter
		self._findStickerPositions()
		self.position = self.faces.copy() # cube is solved by default

		# Unique values of `self.faces`, to be used as distinct color identifiers
		self.faceValues = tuple(sorted(set(self.faces)))

		# Color swaps are specified as tuples having a new color at the index of each old color;
		# I know a real Rubik's cube does not support color swaps, but they are good for elliminating
		# symmetries, making the virtual puzzle smaller.
		self.noColorSwap = np.arange(6)

		# rotation angles available for individual axes
		rotations = [
			self.squareRotations if size[1] == size[2] else (180, ),
			self.squareRotations if size[0] == size[2] else (180, ),
			self.squareRotations if size[0] == size[1] else (180, ),
		]

		# Piece-swapping cube moves (like rotations of a face or of the whole cube);
		# Implemented as arrays of indices specifying how the `position` array should be reordered
		self.moveOrders = {
			noMove: np.arange(len(self.position)),
		}

		# Finding layer (solving) moves
		for axis in range(3):
			for fromBack in (False, True):
				for thickness in range(1, min(size[axis], self.maxThickness+1)):
					for rotation in rotations[axis]:
						move = Move(MoveType.sliceRot, (axis, fromBack, thickness, rotation))
						order = self._makeMoveOrder(axis, fromBack, thickness, rotation)
						self.moveOrders[move] = order

		# Finding whole-cube rotations
		for axis in range(3):
			for rotation in rotations[axis]:
				move = Move(MoveType.cubeRot, (axis, False, size[axis]+1, rotation))
				order = self._makeMoveOrder(axis, False, size[axis]+1, rotation)
				self.moveOrders[move] = order

		# Disabling moves that move the fixed corner
		if fixCorner:
			cornerMask = (np.array(self._centers) <= 1).all(axis=1)
			assert cornerMask.sum() == 3
			self.moveOrders = { move: order
			                    for move, order in self.moveOrders.items()
			                        if (order[cornerMask] == self.moveOrders[noMove][cornerMask]).all() }

		self.solvingMoves = sorted(m for m in self.moveOrders if m.type == MoveType.sliceRot)
		self.rotationMoves = sorted(m for m in self.moveOrders if m.type == MoveType.cubeRot)

		# precalculating things to speed up further operations
		# ?? make static?
		self._rememberedPositions = []
		self.faceIndices = [(self.faces == i) for i in self.faceValues]
		self._findRotations()
		self._findRotationalSwaps()
		self._invertMoveToPosition()
		self._findRotatedMoves()



	def __str__(self):
		size = self.size[0] if len(set(self.size)) == 1 else f"({self.size[0]}, {self.size[1]}, {self.size[2]})"
		args = [f"size={size}"]
		if self.squareRotations != (90, 270):
			args.append(f"squareRotations={self.squareRotations}")
		if self.maxThickness < np.inf:
			args.append(f"maxThickness={self.maxThickness}")
		if self.fixCorner:
			args.append("fixCorner=True")
		if self.stickerFilter:
			args.append(f"stickerFilter={self.stickerFilter}")
		return self.__class__.__name__ + "(" + ", ".join(args) + ")"



	## Pre-calculation methods #####################################################################


	def _findStickerPositions(self):
		"""
		Builds a list of geometrical positions of sticker corners to allow plotting.
		Also finds sticker centers (in different coordinates) for rotation simulations.

		Cube position is represented as a 1-D numpy array of integers, different integers meaning
		different colors. The constant `faces` array specifies which elements of the `position`
		array belong to the same face and therefore should be solved to have the same color (alhough
		not necessarily in the same order as `faces`).
		"""
		stickers = []
		centers = []
		faces = []
		cubeletCoords = []

		X, Y, Z = self.size

		for z in range(Z):
			for y in range(Y):
				stickers.append([[0, y, z], [0, y+1, z], [0, y+1, z+1], [0, y, z+1]])
				centers.append([0, y*2+1, z*2+1])
				cubeletCoords.append([0, y, z])
				faces.append(0)

		for z in range(Z):
			for y in range(Y):
				stickers.append([[X, y, z], [X, y+1, z], [X, y+1, z+1], [X, y, z+1]])
				centers.append([X*2, y*2+1, z*2+1])
				cubeletCoords.append([X-1, y, z])
				faces.append(1)

		for z in range(Z):
			for x in range(X):
				stickers.append([[x, 0, z], [x+1, 0, z], [x+1, 0, z+1], [x, 0, z+1]])
				centers.append([x*2+1, 0, z*2+1])
				cubeletCoords.append([x, 0, z])
				faces.append(2)

		for z in range(Z):
			for x in range(X):
				stickers.append([[x, Y, z], [x+1, Y, z], [x+1, Y, z+1], [x, Y, z+1]])
				centers.append([x*2+1, Y*2, z*2+1])
				cubeletCoords.append([x, Y-1, z])
				faces.append(3)

		for y in range(Y):
			for x in range(X):
				stickers.append([[x, y, 0], [x+1, y, 0], [x+1, y+1, 0], [x, y+1, 0]])
				centers.append([x*2+1, y*2+1, 0])
				cubeletCoords.append([x, y, 0])
				faces.append(4)

		for y in range(Y):
			for x in range(X):
				stickers.append([[x, y, Z], [x+1, y, Z], [x+1, y+1, Z], [x, y+1, Z]])
				centers.append([x*2+1, y*2+1, Z*2])
				cubeletCoords.append([x, y, Z-1])
				faces.append(5)

		self._centers = centers

		self._stickers = np.array(stickers, dtype=np.float64)
		self._stickers[..., 0] -= X/2
		self._stickers[..., 1] -= Y/2
		self._stickers[..., 2] -= Z/2

		self.faces = np.array(faces)

		if self.stickerFilter:
			mask = np.empty(len(centers), dtype=bool)
			for i in range(len(centers)):
				mask[i] = self.stickerFilter(
					cube = self,
					border = self._stickers[i, ...].copy(),
					center = np.array(self._centers[i]),
					face = self.faces[i],
				)
			self._stickers = self._stickers[mask, ...]
			self._centers = [c for c, m in zip(self._centers, mask) if m]
			self.faces = self.faces[mask, ...]

		# assigning squares to cubelets

		cubeletCoords = np.array(cubeletCoords)
		coord2index = {}
		for index, coord in enumerate(cubeletCoords):
			coord = tuple(coord)
			if coord not in coord2index:
				coord2index[coord] = index
		cubelets = [coord2index[tuple(coord)] for coord in cubeletCoords]
		self.cubelets = np.array(cubelets)


	def _findMoving(self, axis, fromBack, thickness, rotation):
		"""
		Finds which stickers are moving during the specified move
		"""
		if fromBack:
			axisThreshold = (self.size[axis]-thickness) * 2
			return [(c[axis] > axisThreshold) for c in self._centers]
		else:
			axisThreshold = thickness * 2
			return [(c[axis] < axisThreshold) for c in self._centers]


	def _makeMoveOrder(self, axis, fromBack, thickness, rotation):
		"""
		Constructs a move order from a geometrical model of the move
		"""
		moving = self._findMoving(axis, fromBack, thickness, rotation)

		# finding sticker centers after rotation
		newCenters = []
		for c, stickerMoving in zip(self._centers, moving):
			if stickerMoving:
				if axis == 0 and rotation == 90:
					newCenters.append([c[0], c[2], 2*self.size[1]-c[1]])
				elif axis == 0 and rotation == 180:
					newCenters.append([c[0], 2*self.size[1]-c[1], 2*self.size[2]-c[2]])
				elif axis == 0 and rotation == 270:
					newCenters.append([c[0], 2*self.size[2]-c[2], c[1]])
				elif axis == 1 and rotation == 90:
					newCenters.append([c[2], c[1], 2*self.size[0]-c[0]])
				elif axis == 1 and rotation == 180:
					newCenters.append([2*self.size[0]-c[0], c[1], 2*self.size[2]-c[2]])
				elif axis == 1 and rotation == 270:
					newCenters.append([2*self.size[2]-c[2], c[1], c[0]])
				elif axis == 2 and rotation == 90:
					newCenters.append([c[1], 2*self.size[0]-c[0], c[2]])
				elif axis == 2 and rotation == 180:
					newCenters.append([2*self.size[0]-c[0], 2*self.size[1]-c[1], c[2]])
				elif axis == 2 and rotation == 270:
					newCenters.append([2*self.size[1]-c[1], c[0], c[2]])
				else:
					raise ValueError(f"Bad move spec: axis {axis} (supported 0, 1, 2), rotation {rotation} (supported 90, 180, 270)")
				if newCenters[-1] not in self._centers:
					breakpoint()
			else:
				newCenters.append(c)

		order = np.empty(len(self._centers), dtype=int)
		for i, c in enumerate(self._centers):
			order[i] = newCenters.index(c)
		return order


	def _findRotations(self):
		"""
		Constructs all cube transformations that can be reached by a finite sequence
		of rotationMoves, to be used as symmetries in solving algorithms.
		"""
		defaultPos = tuple(range(len(self.faces)))
		self.rotationToMoves = { defaultPos: () }
		queue = [(np.array(defaultPos), ())]
		while queue:
			startPos, moveSequence = queue.pop()
			for move in self.rotationMoves:
				rotatedPos = self.applyRotation(self.moveOrders[move], startPos)
				if tuple(rotatedPos) not in self.rotationToMoves:
					queue.insert(0, (rotatedPos, moveSequence + (move,)))
					self.rotationToMoves[tuple(rotatedPos)] = moveSequence + (move,)
		self.rotations = [np.array(pos) for pos in sorted(self.rotationToMoves.keys())]


	def _findRotationalSwaps(self):
		"""
		For each permutation of colors, finds the rotational color swap which
		makes the color permutation lexicographically smallest (= canonical).
		"""
		# finding rotational color swaps
		colorSwaps = { self.faceValues }
		for rot in self.rotations:
			swappedfaces = self.faces[rot]
			mapping = np.stack((self.faces, swappedfaces), axis=0)
			mapping = np.unique(mapping, axis=1)
			assert mapping.shape[1] == len(self.faceValues), \
			       "Cube rotations should not mix colors"
			assert (mapping[0, :] == np.array(self.faceValues)).all(), \
			       "Each color should have something mapped"
			colorSwaps.add(tuple(mapping[1, :]))
		sortedSwaps = sorted(colorSwaps)

		# finding the right swap for each color order
		self._colorOrder2RotationalSwap = {}
		for colorOrder in itertools.permutations(self.faceValues):
			swappedOrders = [tuple(swap[c] for c in colorOrder) for swap in sortedSwaps]
			smallestInd = min(range(len(swappedOrders)), key=lambda i: swappedOrders[i])
			self._colorOrder2RotationalSwap[colorOrder] = np.array(sortedSwaps[smallestInd])


	def _invertMoveToPosition(self):
		"""
		Constructs a mapping from position scrambled by a single move to that move
		"""
		self.rememberPosition()
		defaultPos = tuple(range(len(self.faces)))
		self._position2move = { defaultPos: noMove }
		for move in self.solvingMoves+self.rotationMoves:
			self.setPosition(defaultPos)
			self.move(move)
			self._position2move[tuple(self.getPosition())] = move
		self.restorePosition()

	def _findRotatedMoves(self):
		"""
		For each non-rotation move, finds to which move this transforms when cube is rotated
		"""
		self.rememberPosition()
		self._rotatedMoves = {}
		for move in self.solvingMoves + self.rotationMoves + [noMove]:
			self._rotatedMoves[move] = {}
			for rotation in self.rotations:
				self.position = rotation
				self.move(move)
				self.position = self.applyRotation(self.invertRotation(rotation), self.position)
				pos = tuple(self.position)
				if pos in self._position2move:
					self._rotatedMoves[move][tuple(rotation)] = self._position2move[pos]
		self.restorePosition()

	def rotateMove(self, move, rotation):
		"""
		Maps a move to an equivalent move on a rotated cube
		"""
		if move.type in (MoveType.sliceRot, MoveType.cubeRot):
			return self._rotatedMoves[move][tuple(self.invertRotation(rotation))]
		else:
			return move



	## Accessing cube state ########################################################################


	def getPosition(self):
		return self.position.copy()

	def setPosition(self, position):
		position = np.array(position, dtype=int)
		assert position.shape == self.position.shape
		self.position = position

	def rememberPosition(self, pos=None):
		self._rememberedPositions.append(self.position)
		if pos is not None:
			self.position = np.array(pos)
		return self.position

	def restorePosition(self):
		self.position = self._rememberedPositions[-1]
		self._rememberedPositions[-1:] = []



	## Handling moves ##############################################################################


	@staticmethod
	def applyRotation(newOrder, position):
		return position[newOrder]

	@staticmethod
	def applyColorSwap(swap, position):
		return cy_applyColorSwap(swap, position)

	@staticmethod
	def invertRotation(rotation):
		return np.argsort(rotation)

	@staticmethod
	def invertColorSwap(swap):
		return np.argsort(swap)

	@staticmethod
	def invertMove(move):
		if move.type == MoveType.none:
			return noMove
		if move.type in (MoveType.sliceRot, MoveType.cubeRot):
			axis, fromBack, thickness, angle = move.param
			invAngle = 360 - angle
			return Move(move.type, (axis, fromBack, thickness, invAngle))
		raise NotImplementedError(f"Inverting move of type {move.type}")

	def move(self, move):
		if move.type == MoveType.colorSwap:
			self.position = self.applyColorSwap(move.param, self.position)
		else:
			self.position = self.applyRotation(self.moveOrders[move], self.position)



	## Handling higher-level operations ############################################################


	def reset(self):
		self.position = self.faces.copy()

	def scramble(self, moves=30, seed=None):
		rng = np.random.default_rng(seed)
		for i in range(moves):
			moveInd = rng.integers(len(self.solvingMoves))
			move = self.solvingMoves[moveInd]
			self.move(move)

	def distinct(self):
		self.position = np.arange(len(self.faces))

	def init(self, steps=[], seed=None):
		"""
		Performs a sequence of higher-level actions - right now either "reset" or "scramble"
		"""
		if isinstance(steps, CubeTransform):
			steps = [steps]
		for step in steps:
			if step.method == CubeTransformMethod.reset:
				self.reset(**step.kwargs)
			elif step.method == CubeTransformMethod.scramble:
				kwargs = {**step.kwargs}
				if kwargs.get("seed") is None:
					kwargs["seed"] = seed
				self.scramble(**kwargs)
			elif step.method == CubeTransformMethod.distinct:
				self.distinct(**step.kwargs)
			else:
				raise ValueError(f"Unknown init method: {step.method}")



	## Canonization ################################################################################


	def _getCanonicColorSwap(self, position, mode):
		"""
		Finds the color swap which lexicographically minimizes `position`
		"""
		colorOrder = uniqueColors(position, len(self.faceValues))
		if mode == 1: # color swaps reachable by rotations of solved cube
			# find the best swap ampong precalculated ones
			return self._colorOrder2RotationalSwap[tuple(colorOrder)]
		elif mode == 2: # general color swaps, including those unreachable by valid moves
			# assign 0 to the color which occurs first, 1 to the second one, etc.
			return self.invertColorSwap(colorOrder)
		else:
			raise ValueError(f"Unknown color swap mode: {mode}")


	def _getRotatedPositions(self, position=None):
		"""
		Finds all rotations of `position` (including those by more than 90deg and/or multiple axes)
		"""
		if position is None:
			position = self.getPosition()
		return [(self.applyRotation(rot, position), rot) for rot in self.rotations]


	def _canonizePosition(self, position=None, *, rotationEquivalent=False, colorSwapEquivalent=0):
		"""
		Finds the canonic form of `position` under given equivalence classes.
		Returns the canonic position along with a rotation and color swap which do
		the canonization.
		"""
		if position is None:
			position = self.getPosition()

		if rotationEquivalent:
			rotatedPositions = self._getRotatedPositions(position)
		else:
			rotatedPositions = [(position, self.rotations[0])]

		if colorSwapEquivalent != 0:
			equivalentPositions = []
			for pos, rot in rotatedPositions:
				swap = self._getCanonicColorSwap(pos, mode=colorSwapEquivalent)
				pos = self.applyColorSwap(swap, pos)
				equivalentPositions.append((pos, rot, swap))
		else:
			equivalentPositions = [(pos, rot, self.noColorSwap) for pos, rot in rotatedPositions]

		minIndex = min(range(len(equivalentPositions)),
		               key = lambda i: tuple(equivalentPositions[i][0]))
		return equivalentPositions[minIndex]


	def canonizePosition(self, position=None, **kwargs):
		"""
		Finds the canonic form of `position` under given equivalence classes.
		Returns the canonic position along with a sequence of `Move`s which do
		the canonization.
		"""
		position, canonizingRotation, canonizingColorSwap = self._canonizePosition(
			position, **kwargs)
		canonizingMoves = self.rotationToMoves[tuple(canonizingRotation)]
		if (canonizingColorSwap != self.noColorSwap).any():
			canonizingMoves = canonizingMoves + (Move(MoveType.colorSwap, canonizingColorSwap),)
		return position, canonizingMoves


	def isCanonicPosition(self, position=None, **kwargs):
		"""
		Checks if a position is canonic under given equivalence classes
		"""
		position = self.rememberPosition(position)
		canonicPosition, *_ = self._canonizePosition(position, **kwargs)
		isCanonic = (position == canonicPosition).all()
		self.restorePosition()
		return isCanonic



	## Misc analysis ###############################################################################


	def pos2int(self, posTuple):
		numFaces = len(self.faceValues)
		posInt = 0
		for face in posTuple:
			posInt = posInt*numFaces + int(face)
		return posInt

	def pos2tuple(self, posInt):
		numFaces = len(self.faceValues)
		posTuple = []
		for _ in self.faces:
			posTuple.append(posInt % numFaces)
			posInt //= numFaces
		posTuple.reverse()
		return tuple(posTuple)

	def getPositionsAfter(self,
		numMoves,
		*,
		position = None,
		rotationEquivalent = False,
		colorSwapEquivalent = 0,
	):
		"""
		Finds all positions reachable by a given number of moves.
		Yields a sequence of position sets, n-th set containing positions reached
		by n-1 moves which cannot be reached by any smaller number of moves.
		"""
		if position is None:
			position = self.getPosition()

		positions = { self.pos2int(position) }
		yield positions # after 0 moves

		oldPositions = set(positions)
		for moveInd in range(numMoves):
			newPositions = set()
			for posInt in positions:
				position = np.array(self.pos2tuple(posInt))
				for move in self.solvingMoves:
					newPos, *_ = self._canonizePosition(
						self.applyRotation(self.moveOrders[move], position),
						rotationEquivalent = rotationEquivalent,
						colorSwapEquivalent = colorSwapEquivalent,
					)
					newPosInt = self.pos2int(newPos)
					if newPosInt not in oldPositions:
						newPositions.add(newPosInt)
			positions = newPositions
			yield positions
			oldPositions |= positions


	def isSolved(self, position=None):
		if position is None:
			position = self.getPosition()
		for faceInd in self.faceIndices:
			if len(set(position[faceInd])) != 1:
				return False
		return True

	def getScore(self, position=None):
		if position is None:
			position = self.getPosition()
		return sum(1/len(set(position[faceInd])) for faceInd in self.faceIndices)



	## Drawing #####################################################################################


	# Mapping "color" integers to actual colors (R, G, B)
	colors = [
		(0xDD, 0x44, 0x22),
		(0xFF, 0x88, 0x00),
		(0xEE, 0xEE, 0x00),
		(0xEE, 0xEE, 0xEE),
		(0x44, 0x66, 0xFF),
		(0x44, 0xCC, 0x44),
	]

	def plot(
		self,
		ax = None,
		*,
		colors = None,
		rotMove = None,
		rotPhase = None,
	):
		"""
		Renders the cube using matplotlib.
		`rotMove` and `rotPhase` are used to draw partially rotated cubes
		or cube parts in animations.
		"""
		ax = _getAx(ax)

		
		stickers = self._stickers.copy()

		if rotMove is not None and rotPhase is not None:
			axis, fromBack, thickness, angle = rotMove.param
			moving = self._findMoving(axis, fromBack, thickness, angle)
			swapAxes = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)
			if angle > 180:
				angle -= 360
			rotAngle = angle * rotPhase * np.pi/180
			stickers[moving, :, swapAxes[0]], stickers[moving, :, swapAxes[1]] = (
				  np.cos(rotAngle) * stickers[moving, :, swapAxes[0]]
				+ np.sin(rotAngle) * stickers[moving, :, swapAxes[1]],
				  np.cos(rotAngle) * stickers[moving, :, swapAxes[1]]
				- np.sin(rotAngle) * stickers[moving, :, swapAxes[0]],
			)

		if colors is None:
			colors = self.colors

#		facecolors = ["#DDDDDD"]*24
#		facecolors[ind] = "#FFAA00"

		mplStickers = Poly3DCollection(
			stickers,
#			facecolors = facecolors,
			facecolors = ["#{:02X}{:02X}{:02X}".format(*colors[c])
			              for c in self.position],
#			facecolors = [ # colors for debugging
#				"#660000", "#AA0000", "#FF0000", "#FF4444",
#				"#004400", "#008800", "#00DD00", "#22FF22",
#				"#000088", "#0000CC", "#2222FF", "#6666FF",
#				"#004444", "#008888", "#00CCCC", "#00FFFF",
#				"#660066", "#AA00AA", "#DD00DD", "#FF44FF",
#				"#444400", "#888800", "#CCCC00", "#FFFF00",
#			],
			edgecolors = ("black", ),
			linewidths = 1,
		)
		ax.add_collection3d(mplStickers)
		ax.set_xlim(-self.size[0]/2, self.size[0]/2)
		ax.set_ylim(-self.size[1]/2, self.size[1]/2)
		ax.set_zlim(-self.size[2]/2, self.size[2]/2)
		plt.axis("off")



	@staticmethod
	def _blendColors(colors0, colors1, mu):
		"""
		Interpolates between two RGB colors.
		Used to animate color swap "moves".
		"""
		return [
			(
				round( c0[0]*(1-mu) + c1[0]*mu ),
				round( c0[1]*(1-mu) + c1[1]*mu ),
				round( c0[2]*(1-mu) + c1[2]*mu ),
			)
			for c0, c1 in zip(colors0, colors1)
		]

	def animateMove(self, move, ax=None):
		"""
		Performs a move accompanied by a matplotlib animation
		"""
		ax = _getAx(ax)

		if move.type == MoveType.none:
			plt.pause(0.3)

		elif move.type in (MoveType.sliceRot, MoveType.cubeRot):
			assert move in self.moveOrders, move
			for phase in np.linspace(0, 1, 30):
				ax.clear()
				self.plot(ax, rotMove=move, rotPhase=phase)
				_updatePlot()

		elif move.type == MoveType.colorSwap:
			for step in range(30):
				colors = self._blendColors(
					self.colors,
					[self.colors[c] for c in move.param],
					step/30,
				)
				ax.clear()
				self.plot(ax, colors=colors)
				_updatePlot()

		else:
			raise NotImplementedError(f"Animation of {move.type} move")

		# making the move
		self.move(move)

		# making sure the right position is plotted at the end
		ax.clear()
		self.plot(ax)
		_updatePlot()



	def readEvalPlotLoop(self, ax=None):
		"""
		Displays an interactive cube controlled by keyboard.
		  q       quit
		  i       reset cube to solved state
		  s       scramble
		  LRFBDU  rotate a face by 90deg (each letter controls a different face)
		  L'R'F'  rotate a face by -90deg
		  lrfbdu  rotate two layers of a face by 90deg (each letter controls a different face)
		  l'r'f'  rotate two layers of a face by -90deg
		  XYZ     rotate whole cube by 90deg (each letter controls a different axis)
		  X'Y'Z'  rotate whole cube by -90deg

		Need to press enter after commands. Multiple commands can be entered at once.
		"""
		ax = _getAx(ax)
		self.plot(ax)
		_updatePlot()

		moveRe = re.compile(r"[a-zA-Z]'?")
		code2Move = {
			"L'": Move(MoveType.sliceRot, (0, False, 1,  90)),
			"L":  Move(MoveType.sliceRot, (0, False, 1, 270)),
			"R":  Move(MoveType.sliceRot, (0, True,  1,  90)),
			"R'": Move(MoveType.sliceRot, (0, True,  1, 270)),
			"F":  Move(MoveType.sliceRot, (1, False, 1,  90)),
			"F'": Move(MoveType.sliceRot, (1, False, 1, 270)),
			"B'": Move(MoveType.sliceRot, (1, True,  1,  90)),
			"B":  Move(MoveType.sliceRot, (1, True,  1, 270)),
			"D'": Move(MoveType.sliceRot, (2, False, 1,  90)),
			"D":  Move(MoveType.sliceRot, (2, False, 1, 270)),
			"U":  Move(MoveType.sliceRot, (2, True,  1,  90)),
			"U'": Move(MoveType.sliceRot, (2, True,  1, 270)),
			"l'": Move(MoveType.sliceRot, (0, False, 2,  90)),
			"l":  Move(MoveType.sliceRot, (0, False, 2, 270)),
			"r":  Move(MoveType.sliceRot, (0, True,  2,  90)),
			"r'": Move(MoveType.sliceRot, (0, True,  2, 270)),
			"f":  Move(MoveType.sliceRot, (1, False, 2,  90)),
			"f'": Move(MoveType.sliceRot, (1, False, 2, 270)),
			"b'": Move(MoveType.sliceRot, (1, True,  2,  90)),
			"b":  Move(MoveType.sliceRot, (1, True,  2, 270)),
			"d'": Move(MoveType.sliceRot, (2, False, 2,  90)),
			"d":  Move(MoveType.sliceRot, (2, False, 2, 270)),
			"u":  Move(MoveType.sliceRot, (2, True,  2,  90)),
			"u'": Move(MoveType.sliceRot, (2, True,  2, 270)),
			"X":  Move(MoveType.cubeRot,  (0, False, np.inf,  90)),
			"X'": Move(MoveType.cubeRot,  (0, False, np.inf, 270)),
			"Z":  Move(MoveType.cubeRot,  (1, False, np.inf,  90)),
			"Z'": Move(MoveType.cubeRot,  (1, False, np.inf, 270)),
			"Y":  Move(MoveType.cubeRot,  (2, False, np.inf,  90)),
			"Y'": Move(MoveType.cubeRot,  (2, False, np.inf, 270)),
		}

		stop = False
		while not stop:
			move = input("> ")
			moves = moveRe.findall(move)
			for m in moves:
				if m == "q":
					stop = True
					break
				elif m == "i":
					self.reset()
					self.plot(ax)
					_updatePlot()
				elif m == "s":
					self.scramble()
					self.plot(ax)
					_updatePlot()
				elif m in code2Move and code2Move[m] in self.moveOrders:
					self.animateMove(code2Move[m], ax)
					_updatePlot()
					plt.pause(0.2)
				else:
					print(f"Bad move: {m}")

	# alias for method readEvalPlotLoop
	repl = readEvalPlotLoop



class Rubik_2x2x2:
	"""
	A virtual 2×2×2 Rubik's cube
	"""

	def __init__(self, *, fixCorner=False):
		self.fixCorner = fixCorner

		self._rememberedPositions = []

		# Piece-swapping cube moves (like rotations of a face or of the whole cube);
		# Implemented as arrays of indices specifying how the `position` array should be reordered
		self.moveOrders = {
			noMove:                       np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]),
		# single-face moves (solving moves)
			Move(MoveType.sliceRot,  1):  np.array([ 1, 3, 0, 2, 4, 5, 6, 7,18, 9,16,11,22,13,20,15,12,17,14,19, 8,21,10,23]),
			Move(MoveType.sliceRot, -1):  np.array([ 2, 0, 3, 1, 4, 5, 6, 7,20, 9,22,11,16,13,18,15,10,17, 8,19,14,21,12,23]),
			Move(MoveType.sliceRot,  2):  np.array([ 0, 1, 2, 3, 5, 7, 4, 6, 8,19,10,17,12,23,14,21,16,13,18,15,20, 9,22,11]),
			Move(MoveType.sliceRot, -2):  np.array([ 0, 1, 2, 3, 6, 4, 7, 5, 8,21,10,23,12,17,14,19,16,11,18, 9,20,15,22,13]),
			Move(MoveType.sliceRot,  3):  np.array([17, 1,16, 3,21, 5,20, 7, 9,11, 8,10,12,13,14,15, 4, 6,18,19, 0, 2,22,23]),
			Move(MoveType.sliceRot, -3):  np.array([20, 1,21, 3,16, 5,17, 7,10, 8,11, 9,12,13,14,15, 2, 0,18,19, 6, 4,22,23]),
			Move(MoveType.sliceRot,  4):  np.array([ 0,19, 2,18, 4,23, 6,22, 8, 9,10,11,13,15,12,14,16,17, 5, 7,20,21, 1, 3]),
			Move(MoveType.sliceRot, -4):  np.array([ 0,22, 2,23, 4,18, 6,19, 8, 9,10,11,14,12,15,13,16,17, 3, 1,20,21, 7, 5]),
			Move(MoveType.sliceRot,  5):  np.array([ 9, 8, 2, 3,13,12, 6, 7, 4, 5,10,11, 0, 1,14,15,17,19,16,18,20,21,22,23]),
			Move(MoveType.sliceRot, -5):  np.array([12,13, 2, 3, 8, 9, 6, 7, 1, 0,10,11, 5, 4,14,15,18,16,19,17,20,21,22,23]),
			Move(MoveType.sliceRot,  6):  np.array([ 0, 1,11,10, 4, 5,15,14, 8, 9, 6, 7,12,13, 2, 3,16,17,18,19,21,23,20,22]),
			Move(MoveType.sliceRot, -6):  np.array([ 0, 1,14,15, 4, 5,10,11, 8, 9, 3, 2,12,13, 7, 6,16,17,18,19,22,20,23,21]),
		# whole-cube rotation moves
			Move(MoveType.cubeRot,  12):  np.array([ 1, 3, 0, 2, 5, 7, 4, 6,18,19,16,17,22,23,20,21,12,13,14,15, 8, 9,10,11]),
			Move(MoveType.cubeRot, -12):  np.array([ 2, 0, 3, 1, 6, 4, 7, 5,20,21,22,23,16,17,18,19,10,11, 8, 9,14,15,12,13]),
			Move(MoveType.cubeRot,  34):  np.array([17,19,16,18,21,23,20,22, 9,11, 8,10,13,15,12,14, 4, 6, 5, 7, 0, 2, 1, 3]),
			Move(MoveType.cubeRot, -34):  np.array([20,22,21,23,16,18,17,19,10, 8,11, 9,14,12,15,13, 2, 0, 3, 1, 6, 4, 7, 5]),
			Move(MoveType.cubeRot,  56):  np.array([ 9, 8,11,10,13,12,15,14, 4, 5, 6, 7, 0, 1, 2, 3,17,19,16,18,21,23,20,22]),
			Move(MoveType.cubeRot, -56):  np.array([12,13,14,15, 8, 9,10,11, 1, 0, 3, 2, 5, 4, 7, 6,18,16,19,17,22,20,23,21]),
		}
		if fixCorner: # disabling moves that move the fixed corner
			self.moveOrders = { move: order
			                    for move, order in self.moveOrders.items()
			                        if move.param in (None, 1, -1, 3, -3, 5, -5) }

		self.solvingMoves = sorted(m for m in self.moveOrders if m.type == MoveType.sliceRot)
		self.rotationMoves = sorted(m for m in self.moveOrders if m.type == MoveType.cubeRot)

		# precalculating things to speed up further operations
		# ?? make static
		self.faceIndices = [(self.faces == i) for i in self.faceValues]
		self._findRotations()
		self._findRotationalSwaps()
		self._invertMoveToPosition()
		self._findRotatedMoves()



	# Cube position is represented as a 1-D numpy array of integers, different integers meaning
	# different colors. The constant `faces` array specifies which elements of the `position`
	# array belong to the same face and therefore should be solved to have the same color (alhough
	# not necessarily in the same order as `faces`). Geometric positions of cube pieces are not
	# specified at this point since they are only needed for plotting (and for derivation of moves,
	# which was done by paper and pencil).
	faces = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
	position = faces.copy() # cube is solved by default

	# Unique values of `faces`, to be used as distinct color identifiers
	faceValues = tuple(sorted(set(faces)))

	# Color swaps are specified as tuples having a new color at the index of each old color;
	# I know a real Rubik's cube does not support color swaps, but they are good for elliminating
	# symmetries, making the virtual puzzle smaller.
	noColorSwap = np.arange(6)


	def __str__(self):
		return f"{self.__class__.__name__}(fixCorner={self.fixCorner})"


	## Accessing cube state ########################################################################

	def getPosition(self):
		return self.position.copy()

	def setPosition(self, position):
		position = np.array(position, dtype=int)
		assert position.shape == self.position.shape
		self.position = position

	def rememberPosition(self, pos=None):
		self._rememberedPositions.append(self.position)
		if pos is not None:
			self.position = np.array(pos)
		return self.position

	def restorePosition(self):
		self.position = self._rememberedPositions[-1]
		self._rememberedPositions[-1:] = []


	## Handling moves ##############################################################################

	@staticmethod
	def applyRotation(newOrder, position):
		return position[newOrder]

	@staticmethod
	def applyColorSwap(swap, position):
		return cy_applyColorSwap(swap, position)

	@staticmethod
	def invertRotation(rotation):
		return np.argsort(rotation)

	@staticmethod
	def invertColorSwap(swap):
		return np.argsort(swap)

	@staticmethod
	def invertMove(move):
		if move.type == MoveType.none:
			return noMove
		if move.type in (MoveType.sliceRot, MoveType.cubeRot):
			return Move(move.type, -move.param)
		raise NotImplementedError(f"Inverting move of type {move.type}")

	def move(self, move):
		if move.type == MoveType.colorSwap:
			self.position = self.applyColorSwap(move.param, self.position)
		else:
			self.position = self.applyRotation(self.moveOrders[move], self.position)


	## Handling higher-level operations ############################################################

	def reset(self):
		self.position = self.faces.copy()

	def scramble(self, moves=30, seed=None):
		rng = np.random.default_rng(seed)
		for i in range(moves):
			moveInd = rng.integers(len(self.solvingMoves))
			move = self.solvingMoves[moveInd]
			self.move(move)

	def init(self, steps=[], seed=None):
		"""
		Performs a sequence of higher-level actions - right now either "reset" or "scramble"
		"""
		if isinstance(steps, CubeTransform):
			steps = [steps]
		for step in steps:
			if step.method == CubeTransformMethod.reset:
				self.reset(**step.kwargs)
			elif step.method == CubeTransformMethod.scramble:
				kwargs = {**step.kwargs}
				if kwargs.get("seed") is None:
					kwargs["seed"] = seed
				self.scramble(**kwargs)
			else:
				raise ValueError(f"Unknown init method: {step.method}")


	## Pre-calculation methods #####################################################################


	def _findRotations(self):
		"""
		Constructs all cube transformations that can be reached by a finite sequence
		of rotationMoves, to be used as symmetries in solving algorithms.
		"""
		defaultPos = tuple(range(len(self.faces)))
		self.rotationToMoves = { defaultPos: () }
		queue = [(np.array(defaultPos), ())]
		while queue:
			startPos, moveSequence = queue.pop()
			for move in self.rotationMoves:
				rotatedPos = self.applyRotation(self.moveOrders[move], startPos)
				if tuple(rotatedPos) not in self.rotationToMoves:
					queue.insert(0, (rotatedPos, moveSequence + (move,)))
					self.rotationToMoves[tuple(rotatedPos)] = moveSequence + (move,)
		self.rotations = [np.array(pos) for pos in sorted(self.rotationToMoves.keys())]


	def _findRotationalSwaps(self):
		"""
		For each permutation of colors, finds the rotational color swap which
		makes the color permutation lexicographically smallest (= canonical).
		"""
		# finding rotational color swaps
		colorSwaps = { self.faceValues }
		for rot in self.rotations:
			swappedfaces = self.faces[rot]
			mapping = np.stack((self.faces, swappedfaces), axis=0)
			mapping = np.unique(mapping, axis=1)
			assert mapping.shape[1] == len(self.faceValues), \
			       "Cube rotations should not mix colors"
			assert (mapping[0, :] == np.array(self.faceValues)).all(), \
			       "Each color should have something mapped"
			colorSwaps.add(tuple(mapping[1, :]))
		sortedSwaps = sorted(colorSwaps)

		# finding the right swap for each color order
		self._colorOrder2RotationalSwap = {}
		for colorOrder in itertools.permutations(self.faceValues):
			swappedOrders = [tuple(swap[c] for c in colorOrder) for swap in sortedSwaps]
			smallestInd = min(range(len(swappedOrders)), key=lambda i: swappedOrders[i])
			self._colorOrder2RotationalSwap[colorOrder] = np.array(sortedSwaps[smallestInd])


	def _invertMoveToPosition(self):
		"""
		Constructs a mapping from position scrambled by a single move to that move
		"""
		self.rememberPosition()
		defaultPos = tuple(range(len(self.faces)))
		self._position2move = { defaultPos: noMove }
		for move in self.solvingMoves+self.rotationMoves:
			self.setPosition(defaultPos)
			self.move(move)
			self._position2move[tuple(self.getPosition())] = move
		self.restorePosition()

	def _findRotatedMoves(self):
		"""
		For each non-rotation move, finds to which move this transforms when cube is rotated
		"""
		self.rememberPosition()
		self._rotatedMoves = {}
		for move in self.solvingMoves + self.rotationMoves + [noMove]:
			self._rotatedMoves[move] = {}
			for rotation in self.rotations:
				self.position = rotation
				self.move(move)
				self.position = self.applyRotation(self.invertRotation(rotation), self.position)
				pos = tuple(self.position)
				if pos in self._position2move:
					self._rotatedMoves[move][tuple(rotation)] = self._position2move[pos]
		self.restorePosition()

	def rotateMove(self, move, rotation):
		"""
		Maps a move to an equivalent move on a rotated cube
		"""
		if move.type in (MoveType.sliceRot, MoveType.cubeRot):
			return self._rotatedMoves[move][tuple(self.invertRotation(rotation))]
		else:
			return move


	## Canonization ################################################################################


	def _getCanonicColorSwap(self, position, mode):
		"""
		Finds the color swap which lexicographically minimizes `position`
		"""
		colorOrder = uniqueColors(position, len(self.faceValues))
		if mode == 1: # color swaps reachable by rotations of solved cube
			# find the best swap ampong precalculated ones
			return self._colorOrder2RotationalSwap[tuple(colorOrder)]
		elif mode == 2: # general color swaps, including those unreachable by valid moves
			# assign 0 to the color which occurs first, 1 to the second one, etc.
			return self.invertColorSwap(colorOrder)
		else:
			raise ValueError(f"Unknown color swap mode: {mode}")


	def _getRotatedPositions(self, position=None):
		"""
		Finds all rotations of `position` (including those by more than 90deg and/or multiple axes)
		"""
		if position is None:
			position = self.getPosition()
		return [(self.applyRotation(rot, position), rot) for rot in self.rotations]


	def _canonizePosition(self, position=None, *, rotationEquivalent=False, colorSwapEquivalent=0):
		"""
		Finds the canonic form of `position` under given equivalence classes.
		Returns the canonic position along with a rotation and color swap which do
		the canonization.
		"""
		if position is None:
			position = self.getPosition()

		if rotationEquivalent:
			rotatedPositions = self._getRotatedPositions(position)
		else:
			rotatedPositions = [(position, self.rotations[0])]

		if colorSwapEquivalent != 0:
			equivalentPositions = []
			for pos, rot in rotatedPositions:
				swap = self._getCanonicColorSwap(pos, mode=colorSwapEquivalent)
				pos = self.applyColorSwap(swap, pos)
				equivalentPositions.append((pos, rot, swap))
		else:
			equivalentPositions = [(pos, rot, self.noColorSwap) for pos, rot in rotatedPositions]

		minIndex = min(range(len(equivalentPositions)),
		               key = lambda i: tuple(equivalentPositions[i][0]))
		return equivalentPositions[minIndex]


	def canonizePosition(self, position=None, **kwargs):
		"""
		Finds the canonic form of `position` under given equivalence classes.
		Returns the canonic position along with a sequence of `Move`s which do
		the canonization.
		"""
		position, canonizingRotation, canonizingColorSwap = self._canonizePosition(
			position, **kwargs)
		canonizingMoves = self.rotationToMoves[tuple(canonizingRotation)]
		if (canonizingColorSwap != self.noColorSwap).any():
			canonizingMoves = canonizingMoves + (Move(MoveType.colorSwap, canonizingColorSwap),)
		return position, canonizingMoves


	def isCanonicPosition(self, position=None, **kwargs):
		"""
		Checks if a position is canonic under given equivalence classes
		"""
		position = self.rememberPosition(position)
		canonicPosition, *_ = self._canonizePosition(position, **kwargs)
		isCanonic = (position == canonicPosition).all()
		self.restorePosition()
		return isCanonic



	## Misc analysis ###############################################################################


	def getPositionsAfter(self,
		numMoves,
		*,
		position = None,
		rotationEquivalent = False,
		colorSwapEquivalent = 0,
	):
		"""
		Finds all positions reachable by a given number of moves.
		Yields a sequence of position sets, n-th set containing positions reached
		by n-1 moves which cannot be reached by any smaller number of moves.
		"""
		if position is None:
			position = self.getPosition()

		positions = { tuple(position) }
		yield positions # after 0 moves

		oldPositions = set(positions)
		for moveInd in range(numMoves):
			newPositions = set()
			for position in positions:
				position = np.array(position)
				for move in self.solvingMoves:
					newPos, *_ = self._canonizePosition(
						self.applyRotation(self.moveOrders[move], position),
						rotationEquivalent = rotationEquivalent,
						colorSwapEquivalent = colorSwapEquivalent,
					)
					newPositions.add(tuple(newPos))
			positions = newPositions - oldPositions
			yield positions
			oldPositions |= positions


	def isSolved(self, position=None):
		if position is None:
			position = self.getPosition()
		for faceInd in self.faceIndices:
			if len(set(position[faceInd])) != 1:
				return False
		return True

	def getScore(self, position=None):
		if position is None:
			position = self.getPosition()
		return sum(1/len(set(position[faceInd])) for faceInd in self.faceIndices)

	minScore = 1.5
	maxScore = 6.0



	## Drawing #####################################################################################


	# Mapping "color" integers to actual colors (R, G, B)
	colors = [
		(0xDD, 0x44, 0x22),
		(0xFF, 0x88, 0x00),
		(0xEE, 0xEE, 0x00),
		(0xEE, 0xEE, 0xEE),
		(0x44, 0x66, 0xFF),
		(0x44, 0xCC, 0x44),
	]

	# Identifying which stickers are affected by full and partial rotation movements
	# and along which axis they rotate to allow drawing them partially rotated
	_sliceInds = {
		1: ([0, 1, 2, 3,  8, 10, 12, 14, 16, 18, 20, 22], [1, 2]),
		2: ([4, 5, 6, 7,  9, 11, 13, 15, 17, 19, 21, 23], [1, 2]),
		3: ([0, 2, 4, 6,  8,  9, 10, 11, 16, 17, 20, 21], [0, 2]),
		4: ([1, 3, 5, 7, 12, 13, 14, 15, 18, 19, 22, 23], [0, 2]),
		5: ([0, 1, 4, 5,  8,  9, 12, 13, 16, 17, 18, 19], [0, 1]),
		6: ([2, 3, 6, 7, 10, 11, 14, 15, 20, 21, 22, 23], [0, 1]),
		12: (list(range(24)), [1, 2]),
		34: (list(range(24)), [0, 2]),
		56: (list(range(24)), [0, 1]),
	}

	_stickers = np.asarray([

		[[0,0,0], [0,1,0], [0,1,1], [0,0,1]],  # a sticker   #
		[[0,1,0], [0,2,0], [0,2,1], [0,1,1]],                # a face
		[[0,0,1], [0,1,1], [0,1,2], [0,0,2]],                #
		[[0,1,1], [0,2,1], [0,2,2], [0,1,2]],                #

		[[2,0,0], [2,1,0], [2,1,1], [2,0,1]],
		[[2,1,0], [2,2,0], [2,2,1], [2,1,1]],
		[[2,0,1], [2,1,1], [2,1,2], [2,0,2]],
		[[2,1,1], [2,2,1], [2,2,2], [2,1,2]],

		[[0,0,0], [1,0,0], [1,0,1], [0,0,1]],
		[[1,0,0], [2,0,0], [2,0,1], [1,0,1]],
		[[0,0,1], [1,0,1], [1,0,2], [0,0,2]],
		[[1,0,1], [2,0,1], [2,0,2], [1,0,2]],

		[[0,2,0], [1,2,0], [1,2,1], [0,2,1]],
		[[1,2,0], [2,2,0], [2,2,1], [1,2,1]],
		[[0,2,1], [1,2,1], [1,2,2], [0,2,2]],
		[[1,2,1], [2,2,1], [2,2,2], [1,2,2]],

		[[0,0,0], [1,0,0], [1,1,0], [0,1,0]],
		[[1,0,0], [2,0,0], [2,1,0], [1,1,0]],
		[[0,1,0], [1,1,0], [1,2,0], [0,2,0]],
		[[1,1,0], [2,1,0], [2,2,0], [1,2,0]],

		[[0,0,2], [1,0,2], [1,1,2], [0,1,2]],
		[[1,0,2], [2,0,2], [2,1,2], [1,1,2]],
		[[0,1,2], [1,1,2], [1,2,2], [0,2,2]],
		[[1,1,2], [2,1,2], [2,2,2], [1,2,2]],

	], dtype=float) - 1 # shifting to center from [0, 2] interval


	def plot(
		self,
		ax = None,
		*,
		colors = None,
		rotAxis = None,
		rotAngle = None,
	):
		"""
		Renders the cube using matplotlib.
		`rotAxis` and `rotAngle` are used to draw partially rotated cubes
		or cube parts in animations.
		"""
		ax = _getAx(ax)

		
		stickers = self._stickers.copy()

		if rotAxis is not None and rotAngle is not None:
			layer, axes = self._sliceInds[rotAxis]
			stickers[layer, :, axes[0]], stickers[layer, :, axes[1]] = (
				  np.cos(rotAngle) * stickers[layer, :, axes[0]]
				+ np.sin(rotAngle) * stickers[layer, :, axes[1]],
				  np.cos(rotAngle) * stickers[layer, :, axes[1]]
				- np.sin(rotAngle) * stickers[layer, :, axes[0]],
			)

		if colors is None:
			colors = self.colors

		mplStickers = Poly3DCollection(
			stickers,
			facecolors = ["#{:02X}{:02X}{:02X}".format(*colors[c])
			              for c in self.position],
#			facecolors = [ # colors for debugging
#				"#660000", "#AA0000", "#FF0000", "#FF4444",
#				"#004400", "#008800", "#00DD00", "#22FF22",
#				"#000088", "#0000CC", "#2222FF", "#6666FF",
#				"#004444", "#008888", "#00CCCC", "#00FFFF",
#				"#660066", "#AA00AA", "#DD00DD", "#FF44FF",
#				"#444400", "#888800", "#CCCC00", "#FFFF00",
#			],
			edgecolors = ("black", ),
			linewidths = 1,
		)
		ax.add_collection3d(mplStickers)
		ax.set_xlim(-1, 1)
		ax.set_ylim(-1, 1)
		ax.set_zlim(-1, 1)
		plt.axis("off")



	@staticmethod
	def _blendColors(colors0, colors1, mu):
		"""
		Interpolates between two RGB colors.
		Used to animate color swap "moves".
		"""
		return [
			(
				round( c0[0]*(1-mu) + c1[0]*mu ),
				round( c0[1]*(1-mu) + c1[1]*mu ),
				round( c0[2]*(1-mu) + c1[2]*mu ),
			)
			for c0, c1 in zip(colors0, colors1)
		]

	def animateMove(self, move, ax=None):
		"""
		Performs a move accompanied by a matplotlib animation
		"""
		ax = _getAx(ax)

		if move.type == MoveType.none:
			plt.pause(0.3)

		elif move.type in (MoveType.sliceRot, MoveType.cubeRot):
			assert move in self.moveOrders, move
			rotAxis = abs(move.param)
			rotDirection = 1 if move.param >= 0 else -1
			for angle in np.linspace(0, np.pi/2, 30):
				ax.clear()
				self.plot(ax, rotAxis=rotAxis, rotAngle=angle*rotDirection)
				_updatePlot()

		elif move.type == MoveType.colorSwap:
			for step in range(30):
				colors = self._blendColors(
					self.colors,
					[self.colors[c] for c in move.param],
					step/30,
				)
				ax.clear()
				self.plot(ax, colors=colors)
				_updatePlot()

		else:
			raise NotImplementedError(f"Animation of {move.type} move")

		# making the move
		self.move(move)

		# making sure the right position is plotted at the end
		ax.clear()
		self.plot(ax)
		_updatePlot()



	def readEvalPlotLoop(self, ax=None):
		"""
		Displays an interactive cube controlled by keyboard.
		  q       quit
		  r       reset cube to solved state
		  s       scramble
		  LRFBDU  rotate a face by 90deg (each letter controls a different face)
		  L'R'F'  rotate a face by -90deg
		  XYZ     rotate the cube by 90deg (each letter controls a different axis)
		  X'Y'Z'  rotate the cube by -90deg

		Need to press enter after commands. Multiple commands can be entered at once.
		"""
		ax = _getAx(ax)
		self.plot(ax)
		_updatePlot()

		moveRe = re.compile(r"\S'?")
		code2Move = {
			"L'": Move(MoveType.sliceRot, 1),  "L":  Move(MoveType.sliceRot, -1),
			"R":  Move(MoveType.sliceRot, 2),  "R'": Move(MoveType.sliceRot, -2),
			"F":  Move(MoveType.sliceRot, 3),  "F'": Move(MoveType.sliceRot, -3),
			"B'": Move(MoveType.sliceRot, 4),  "B":  Move(MoveType.sliceRot, -4),
			"D'": Move(MoveType.sliceRot, 5),  "D":  Move(MoveType.sliceRot, -5),
			"U":  Move(MoveType.sliceRot, 6),  "U'": Move(MoveType.sliceRot, -6),
			"X":  Move(MoveType.cubeRot, 12),  "X'": Move(MoveType.cubeRot, -12),
			"Z":  Move(MoveType.cubeRot, 34),  "Z'": Move(MoveType.cubeRot, -34),
			"Y":  Move(MoveType.cubeRot, 56),  "Y'": Move(MoveType.cubeRot, -56),
		}

		stop = False
		while not stop:
			move = input("> ")
			moves = moveRe.findall(move)
			for m in moves:
				if m == "q":
					stop = True
					break
				elif m == "r":
					self.reset()
					self.plot(ax)
					_updatePlot()
				elif m == "s":
					self.scramble()
					self.plot(ax)
					_updatePlot()
				elif m in code2Move:
					self.animateMove(code2Move[m], ax)
					_updatePlot()
					plt.pause(0.2)
				else:
					print(f"Bad move: {m}")

	# alias for method readEvalPlotLoop
	repl = readEvalPlotLoop

