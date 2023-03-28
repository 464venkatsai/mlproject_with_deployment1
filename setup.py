from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirement
    '''
    # requirements = []
    with open (file_path) as file:
        requirements = file.readlines()
        requirements = [i.replace('\n','')for i in requirements]
        requirements.remove('-e .')

    return requirements


setup(
   name='MlProject' ,
   version='0.0.1',
   author='VenkatSai',
   author_email='venkatsaikilli464@gmail.com',
   packages=find_packages(),
   # install_requires = ['pandas','numpy','seaborn']
   install_requires = get_requirements('requirement.txt')
)