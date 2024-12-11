from setuptools import setup, find_packages

setup(
    name="bike-rental-prediction",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        "matplotlib",
        "seaborn",
        "PyYAML"
    ]
)