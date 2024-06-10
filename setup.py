from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='heartbd',
      version="0.0.1",
      description="Heart Beat Decoder (api_classification)",
      packages = ['heartbd'])
