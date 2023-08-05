from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hashwise',
    version='0.0.5',
    description='A secure python library for GPU accelerated hashing.',
    long_description=long_description,
    long_description_content_type='text/markdown', 
    author='Anish Kanthamneni',
    packages=find_packages(),
    package_data={
        '': ['*.dll'],
        'hashwise': ['c-libraries/*.dll', 'cuda-libraries/*.dll', 'device-info/*.dll'],
    },
    author_email='akneni@gmail.com',
    install_requires=[
        'tqdm',
        'pathlib'
    ],
)
