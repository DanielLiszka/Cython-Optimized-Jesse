import numpy as np
cimport numpy as np 
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
np.import_array()
import cython
from libc.float cimport DECIMAL_DIG
from libc.time cimport time,time_t
from libc.time cimport tm, mktime
import datetime
# def decimal_sub(double a, double b):
    # return DECIMAL_DIG(a) - DECIMAL_DIG(b)
    
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def arr_equal(np.ndarray[DTYPE_t,ndim=2] a1, np.ndarray[DTYPE_t,ndim=2] a2):
    # if a1.shape[0] != a2.shape[0]:
        # return False
    # cdef size_t rows, columns
    # rows = a1.shape[0]
    # columns = a1.shape[1] 
    # for i in range(rows,-1,-1):
        # for j in range(columns):
            # if a1[i,j] != a2[i,j]:
                # return False 
                # break 
    # return True
    
# def cython_max(np.ndarray[DTYPE_t,ndim=1] y):
    # cdef double min = np.inf
    # cdef double max = -np.inf
    # cdef int i
    # for i in range(y.shape[0]):
        # if y[i] > max:
            # max = y[i]
    # return max     
    
# from cython cimport boundscheck,wraparound
# from cython cimport view

# def  arrayadd(double [::1] a1):
    # cdef int rows 
    # cdef double sum1
    # rows = a1.shape[0]
    # for i in range(rows):
        # sum1 += a1[i]
    # return sum1
# cdef tm time_tuple = {
    # 'tm_sec': second,
    # 'tm_min': minute,
    # 'tm_hour': hour,
    # 'tm_mday': day,
    # 'tm_mon': month - 1,
    # 'tm_year': year - 1900,
    # 'tm_wday': day_of_week,
    # 'tm_yday': day_in_year,
    # 'tm_isdst': dst,
    # 'tm_zone': NULL,
    # 'tm_gmtoff': 0,
# }
# unix_time = mktime(&time_tuple)

def time_func():
    cdef time_t t 
    t = time(NULL)
    return t 
    

# @boundscheck(False)
# @wraparound(False)
# def add_arrays_Cython(double[:,::1] Aarr, double[:,::1] Barr):
    # cdef size_t I,J,i,j
    # I = Barr.shape[0]
    # J = Barr.shape[0]
    # result_as_array = view.array(shape=(I,J), itemsize=sizeof(double), format='i')
    # cdef double[:,::1] Carr = result_as_array
    # for i in range(I):
        # for j in range(J):  
            # Carr[i,j] = Aarr[i,j]+Barr[i,j]
    # return result_as_array