import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='scifAI',
    version='0.0.1',
    description='Open source package for Imaging FLow cytometry and AI',
    author='Ali Boushehri, Aleksandra Kornivetc',
    author_email='ali.boushehri@roche.com',
    license='MIT',
    keywords='Imaging FLow cytometry AI',
    url='https://github.com/aliechoes/scifAI',
    packages=find_packages(exclude=['doc*', 'test*']),
    install_requires=[  "numpy",
                        "pandas",
                        "scikit-learn",
                        "scikit-image",
                        "xgboost",
                        "torch",
                        "torchvision",
                        "skorch"],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
    ],
)