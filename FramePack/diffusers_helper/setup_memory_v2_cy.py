from pathlib import Path
from setuptools import Extension, setup
from Cython.Build import cythonize

root = Path(__file__).parent
source = root / "memory_v2_cy.pyx"

extensions = [
    Extension(
        name="diffusers_helper.memory_v2_cy",
        sources=[str(source)],
        language="c++",
    )
]

setup(
    name="memory_v2_cy",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "boundscheck": False, "wraparound": False},
    ),
)
