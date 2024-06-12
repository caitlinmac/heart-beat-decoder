#instantiate python version
FROM python:3.10.6-buster

# copy these files into the docker container
COPY heartbd heartbd
COPY requirements.txt requirements.txt
COPY setup.py setup.py

# install requirements
RUN pip install -e .
RUN pip install -r requirements.txt

# run container locally
CMD uvicorn heartbd.package_folder.api_file:app --reload 

# run container deployed
# CMD uvicorn heartbd.package_folder.api_file:app --reload —host 0.0.0.0 —port $PORT
