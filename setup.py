from typing import List
from setuptools import setup, find_packages

HYPEN_E_DOT='-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    Retrieve a list of requirements from a file.

    Args:
        file_path (str): The path to the requirements file.

    Returns:
        List[str]: A list of requirements.

    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='tennis-cv',
    version='0.0.1',
    author='Ryan Tri',
    author_email='ry4ntr1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
