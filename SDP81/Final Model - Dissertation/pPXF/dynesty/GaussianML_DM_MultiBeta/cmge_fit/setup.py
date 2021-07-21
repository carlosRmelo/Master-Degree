from distutils.core import setup, Extension
from Cython.Build import cythonize

ext=Extension("mge1d_fit",
              sources=["mge1d_fit.pyx"],
              library_dirs=['/home/carlos/Desktop/pPXF(SDP)/Final/dynesty/GaussianML_DM_MultiBeta/cmge1d_fit'],
              libraries=['mge1d_mpfit']
             )
setup(name='mge1d_fit',
      ext_modules=cythonize([ext]))
