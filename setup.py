"""
Setup script for Arbitrary Numbers
=================================

Python implementation of Arbitrary Numbers for exact symbolic computation
with GPU acceleration, designed for inference models and scientific computing.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Arbitrary Numbers: Exact Symbolic Computation for Python"

# Read version from __init__.py
def get_version():
    init_path = os.path.join('arbitrary_numbers', '__init__.py')
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

setup(
    name='arbitrary-numbers',
    version=get_version(),
    author='Arbitrary Numbers Development Team',
    author_email='dev@arbitrary-numbers.org',
    description='Exact symbolic computation for Python with GPU acceleration',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/arbitrary-number/arbitrary-number',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'torch>=1.12.0',
        'fractions',  # Built-in module
        'decimal',    # Built-in module
    ],
    extras_require={
        'gpu': [
            'cupy-cuda11x>=10.0.0',  # For CUDA 11.x
            # 'cupy-cuda12x>=12.0.0',  # Alternative for CUDA 12.x
        ],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
        'ml': [
            'torch>=1.12.0',
            'torchvision>=0.13.0',
            'jax>=0.3.0',
            'jaxlib>=0.3.0',
        ],
        'all': [
            'cupy-cuda11x>=10.0.0',
            'torch>=1.12.0',
            'torchvision>=0.13.0',
            'jax>=0.3.0',
            'jaxlib>=0.3.0',
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'arbitrary-numbers-demo=examples.inference_model_demo:main',
            'arbitrary-numbers-test=tests.test_basic_functionality:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/arbitrary-number/arbitrary-number/issues',
        'Source': 'https://github.com/arbitrary-number/arbitrary-number',
        'Documentation': 'https://arbitrary-numbers.readthedocs.io/',
    },
    keywords=[
        'symbolic computation',
        'exact arithmetic',
        'rational numbers',
        'gpu acceleration',
        'machine learning',
        'inference models',
        'explainable ai',
        'precision',
        'mathematics',
        'cuda',
        'pytorch'
    ],
    include_package_data=True,
    zip_safe=False,
)
