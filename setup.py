from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            "lj_test.pyx",
            "particle_sim/experiments.pyx",
            "particle_sim/forces.pyx",
            "particle_sim/integrators.pyx"
        ]
    )
)
