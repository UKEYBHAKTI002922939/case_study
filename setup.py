from setuptools import setup, find_packages
from typing import List


REQUIREMENTS_FILE_NAME = "requirements.txt"
# Requirements.txt file open read it
HYPHEN_E_DOT = "-e ."
# Requriments.txt file open
# read
# \n ""
def get_requirements_list()->List[str]:
    with open(REQUIREMENTS_FILE_NAME) as requriment_file:
        requriment_list = requriment_file.readlines()
        requriment_list = [requriment_name.replace("\n", "") for requriment_name in requriment_list]

        if HYPHEN_E_DOT in requriment_list:
            requriment_list.remove(HYPHEN_E_DOT)

        return requriment_list
setup(name='bhakti_study',
      version='0.1',
      packages=find_packages(),
      install_requires = []
     )