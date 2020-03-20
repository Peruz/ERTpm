import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="ERTpm",
    version="0.0.1",
    author="Luca Peruzzo",
    author_email="lperuzzo@lbl.gov",
    description="Python package for managing ERT processing, inversion, and visualization at the project level",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://Peruz.github.io",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: GNU Lesser General Public License " + "v3 (LGPLv3)",
                 "Intended Audience :: Science/Research",
                 "Operating System :: OS Independent"],
    python_requires='>=3.6',
    install_requires=["numpy", "pandas", "pyvista", "numba"]
)
