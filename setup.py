from setuptools import setup, find_packages

setup(
    name='permutenm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    author='Glen Koundry',
    author_email='gkoundry@gmail.com',
    description='Python implementation of Geometric Nelder-Mead Algorithm for the permutation representation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gkoundry/PermuteNM',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Match this with your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
)
