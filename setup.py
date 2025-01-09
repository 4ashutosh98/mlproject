from setuptools import find_packages, setup
from typing import List

HYHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return list the requirements
    """
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYHEN_E_DOT in requirements:
            requirements.remove(HYHEN_E_DOT)


setup(
    name='mlproject',
    version = '0.0.1',
    author = 'Ashutosh Choudhari',
    author_email = '4ashutosh98@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)