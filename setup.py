from setuptools import setup, find_packages
from pathlib import Path

# Read version
version = {}
with open("astro_amase/version.py") as f:
    exec(f.read(), version)

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="astro_amase",
    version=version['__version__'],
    author="Zachary Fried",
    author_email="zfried@mit.edu",
    description="Automated Molecular Assignment and Source Parameter Estimation for Radio Astronomy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/astro_amase",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "rdkit>=2022.9.1",
        "scipy>=1.7.0",
        "bokeh>=2.4.0",
        "numba>=0.54.0",
        "astropy>=4.3.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "astrochem_embedding>=0.2.0",
        "group-selfies @ git+https://github.com/aspuru-guzik-group/group-selfies.git",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'astro-amase=astro_amase.main:cli_entry',
        ],
    },
    include_package_data=True,
    package_data={
        'astro_amase': ['examples/*.yaml'],
    },
)