# python .\setup2.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
import numpy

for file in [
    "geometry.pyx",
    "vessel_track.pyx",
    "structures.pyx",
    'mycc3d_scipy.pyx',
    'plot_3d.pyx'
]:
    setup(
        ext_modules = cythonize(file),
        include_dirs=[numpy.get_include()],
    )