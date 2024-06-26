from setuptools import setup, find_packages
import sys
from pathlib import Path


parent_dir = Path('.').resolve().parent

print(sys.path)

if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
    
import config as config

setup(
    name='Tennis-Analysis-CV',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt", "r").readlines() if line.strip() and not line.strip().startswith('#')
    ],
    author='Ryan Tri',
    author_email='ry4ntr1@gmail.com',
    description='Lorem Ipsum',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ry4ntr1/tennis-cv', 
    python_requires='>=3.11',  
    include_package_data=True,
    zip_safe=False
)
