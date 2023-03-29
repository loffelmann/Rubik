
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
def uniqueColors(long[:] position, int numColors):
	cdef int i, numSeen
	cdef bint *seen = <bint *> malloc(numColors * sizeof(bint))
	for i in range(numColors):
		seen[i] = 0
	cdef np.ndarray[np.int64_t, ndim=1] order = np.empty(numColors, dtype=np.int64)
	numSeen = 0
	for i in range(len(position)):
		if not seen[position[i]]:
			seen[position[i]] = 1
			order[numSeen] = position[i]
			numSeen += 1
			if numSeen == numColors:
				break
	free(seen)
	return order


@cython.boundscheck(False)
def applyColorSwap(long[:] swap, long[:] position):
	cdef int length = len(position)
	cdef np.ndarray[np.int64_t, ndim=1] swapped = np.empty(length, dtype=np.int64)
	cdef int i
	for i in range(length):
		swapped[i] = swap[position[i]]
	return swapped

