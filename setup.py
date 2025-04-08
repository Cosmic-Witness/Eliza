from setuptools import setup, find_packages
from pathlib import Path

long_desc = Path("README.md").read_text()

setup(
  name="ElizaCosmicWitness",  # Fixed the name here
  version="0.1",
  description="A trading assistant powered by CAPM",
  long_description=long_desc,
  long_description_content_type="text/markdown",
  url="https://github.com/yourusername/Eliza",
  author="Ismail",
  author_email="you@example.com",
  license="MIT",
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  packages=find_packages(),
  install_requires=[ "numpy","pandas","colorama","yfinance","statsmodels" ],
  project_urls={
    "Source": "https://github.com/yourusername/Eliza",
    "Tracker": "https://github.com/yourusername/Eliza/issues",
  },
)
