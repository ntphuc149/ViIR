name: ViIR
version: 0.1.0
description: ViIR: The unified framework for fine-tuning Vietnamese Information Retrieval models with various tuning statergies.
author: Truong-Phuc Nguyen
author_email: nguyentruongphuc_12421TN@utehy.edu.vn
url: https://github.com/ntphuc149/ViIR

classifiers:
  - Programming Language :: Python :: 3
  - Topic :: Scientific/Engineering :: Artificial Intelligence
  - Intended Audience :: Science/Research

requires-python: ">=3.8"

dependencies:
  - torch>=1.10.0
  - transformers>=4.16.0
  - sentence-transformers>=2.2.0
  - scikit-learn>=1.0.0
  - pandas>=1.3.0
  - numpy>=1.20.0
  - tqdm>=4.62.0
  - pyyaml>=6.0

dev-dependencies:
  - pytest>=7.0.0
  - black>=22.0.0
  - isort>=5.10.0
  - flake8>=4.0.0
  - mypy>=0.950

[options]
packages = find:
python_requires = >=3.8