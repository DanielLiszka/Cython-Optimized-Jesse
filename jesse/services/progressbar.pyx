import os
# os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"]="0"
from time import time
import numpy as np
cimport numpy as np 
# from jesse.helpers import np_shift
np.import_array()
# from jesse.libs import DynamicNumpyArray
DTYPE = np.float32
   
class Progressbar:
    def __init__(self,int length, int index = 0, int step=1):
        self.length = length
        self.index = index
        
        # validation
        if self.length <= self.index:
            raise ValueError('length must be greater than 0')

        self._time = time()
        self._execution_times = DynamicNumpyArray((3, 1), drop_at=3)
        self.step = step
        self.is_finished = False

    def update(self):
        if not self.is_finished:
            self.index += self.step
            if self.index == self.length:
                self.is_finished = True
        now = time()
        self._execution_times.append(np.array([now - self._time]))
        self._time = now

    @property
    def current(self):
        if self.is_finished:
            return 100
        return round(self.index / self.length * 100, 1)

    @property
    def average_execution_seconds(self):
        return (self._execution_times.array[0:self._execution_times.index+1]).mean()

    @property
    def remaining_index(self):
        if self.is_finished:
            return 0
        return self.length - self.index

    @property
    def estimated_remaining_seconds(self):
        if self.is_finished:
            return 0
        return self.average_execution_seconds * self.remaining_index / self.step

    def finish(self):
        self.is_finished = True
        

class DynamicNumpyArray:
    def __init__(self, shape: tuple,int drop_at = 3,int index = -1, attributes: dict = None):
        self.index = index
        self.array = np.zeros((shape),dtype=DTYPE)
        self.bucket_size = shape[0]
        self.shape = shape
        self.drop_at = drop_at

        
    def __setitem__(self, int i, np.ndarray item):
        cdef Py_ssize_t index = self.index
        # if i < 0:
            # i = (index + 1) - abs(i)
        self.array[self.index] = item
        
    def append(self, np.ndarray item) -> None:
        self.index += 1
        cdef Py_ssize_t index = self.index 
        cdef int shift_num
        # expand if the arr is almost full
        if index != 0 and (index + 1) % self.bucket_size == 0:
            new_bucket = np.zeros(self.shape,dtype=DTYPE)
            self.array = np.concatenate((self.array, new_bucket), axis=0, dtype=DTYPE)
                # drop N% of the beginning values to free memory
        if (index != 0
            and (index + 1) % self.drop_at == 0
        ):
            shift_num = int(self.drop_at / 2)
            self.index -= shift_num
            self.array = c_np_shift(self.array, -shift_num)

        self.array[self.index] = item

def c_np_shift(arr: np.ndarray,int num, int fill_value=0) :
    result = np.empty_like(arr)
    result[num:] = fill_value
    result[:num] = arr[-num:]
    return result