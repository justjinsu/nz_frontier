from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nz_frontier",
    version="0.1.0",
    author="Jinsu Park",
    author_email="jinsu.park@planit.institute",
    description="Portfolio Theory for Corporate Decarbonization: A Risk-Efficiency Framework for Net-Zero Investment under Uncertainty",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jinsupark4/nz_frontier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "streamlit>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    keywords=[
        "portfolio optimization",
        "decarbonization",
        "climate finance",
        "net-zero",
        "real options",
        "markowitz",
        "transition risk",
    ],
)
