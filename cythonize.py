from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import multiprocessing
import os   
import numpy 

#Command to execute: python3 cythonize.py build_ext -j 8 --inplace 

#Note: you will need to point to the Python C extension. For linux, it should be '/usr/include/(python version)/python.h'

#AVX512 CFLAGS arguments: -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -mprefer-vector-width=512

os.environ['CC'] = 'clang'    
os.environ['LDFLAGS'] = '-fuse-ld=lld'
os.environ['LDSHARED'] = 'clang -shared'
os.environ['CFLAGS'] = '-O3 -fuse-ld=lld -fno-openmp -march=native -Wall -mavx -flto=full -DUSE_XSIMD  -I /usr/include/python3.11/Python.h'

Options.annotate = False
Options.convert_range = True
Options.cache_builtins = True
Options.cimport_from_pyx = True 

List = [ 
    "jesse/exceptions/__init__.pyx",
    "jesse/helpers.pyx",
    "jesse/utils.pyx",
    "jesse/store/*.pyx",
    "jesse/services/*.pyx",
    "jesse/strategies/*.pyx",
    "jesse/libs/dynamic_numpy_array/*.pyx",
    "jesse/modes/backtest_mode.pyx",
    "jesse/modes/utils.pyx",
    "jesse/modes/optimize_mode/*.pyx",
    "jesse/enums/*.pyx",
    "jesse/models/*.pyx",
    "jesse/routes/*.pyx",
    "jesse/exchanges/*.pyx",
    "jesse/exchanges/sandbox/*.pyx",
    "jesse/research/*.pyx",
]
exclusions = ["jesse/__init__.py", "jesse/services/web.pyx"] 

setup(
    ext_modules=cythonize( List,nthreads=8, force=True, exclude=exclusions,compiler_directives={'language_level':3,'profile':False,
    'linetrace':False,'binding':False,'infer_types':True,'nonecheck':False,'optimize.use_switch':True,'optimize.unpack_method_calls':True,
    'initializedcheck':False, 'overflowcheck':False, 'overflowcheck.fold': False, 'cdivision_warnings':True, 'cdivision':True,'wraparound':False,
    'boundscheck':False,}), include_dirs=[numpy.get_include(), pythran.get_include()]
)
