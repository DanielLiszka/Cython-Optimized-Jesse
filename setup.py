from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import glob 

# also change in version.py
VERSION = '0.45.0'
DESCRIPTION = "A trading framework for cryptocurrencies"
with open("requirements.txt", "r", encoding="utf-8") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

cython_files = [
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

expanded_files = [f for pattern in cython_files for f in glob.glob(pattern, recursive=True)]
filtered_files = [f for f in expanded_files if f not in exclusions]
cython_extensions = [Extension(f.replace('/', '.').rsplit('.', 1)[0], [f]) for f in filtered_files]


compiler_options = {
    'language_level': 3,
    'profile': False,
    'linetrace': False,
    'binding': False,
    'infer_types': True,
    'nonecheck': False,
    'optimize.use_switch': True,
    'optimize.unpack_method_calls': True,
    'initializedcheck': False,
    'overflowcheck': False,
    'overflowcheck.fold': False,
    'cdivision_warnings': True,
    'cdivision': True,
    'wraparound': False,
    'boundscheck': False
}

setup(
    name='cythonized_jesse',
    version=VERSION,
    author="Saleh Mir",
    author_email="saleh@jesse.trade",
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://jesse.trade",
    project_urls={
        'Documentation': 'https://docs.jesse.trade',
        'Say Thanks!': 'https://jesse.trade/discord',
        'Source': 'https://github.com/jesse-ai/jesse',
        'Tracker': 'https://github.com/jesse-ai/jesse/issues',
    },
    install_requires=REQUIRED_PACKAGES,
    entry_points='''
        [console_scripts]
        cythonized_jesse=cythonized_jesse.__init__:cli
    ''',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    ext_modules=cythonize(cython_extensions, compiler_directives=compiler_options),
    include_dirs=[numpy.get_include()]
)
