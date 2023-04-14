from setuptools import setup

setup(
    name='pandas-join-optimizer',
    version='0.1',
    description='Enhance Pandas join execution with optimizations from database theory',
    author='Alexander Steindl',
    license='MIT',
    py_modules=['pd_join_optimizer'],
    install_requires=[
       'pandas>=1.5.3',
       'networkx>=3.0'
    ],
    python_requires='>=3.9'
)