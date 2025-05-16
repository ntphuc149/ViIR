from setuptools import setup, find_packages

setup(
    name="viir",
    version="0.1.0",
    description="ViIR: The Unified Framework for Fine-tuning Vietnamese Information Retrieval Models with Various Tuning Strategies.",
    author="Truong-Phuc Nguyen",
    author_email="nguyentruongphuc_12421TN@utehy.edu.vn",
    url="https://github.com/ntphuc149/ViIR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research"
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.16.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950"
        ]
    }
)