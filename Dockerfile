FROM python:3.10.6

# what are we going to port to the docker?
COPY heartbd heartbd
COPY requirements.txt requirements.txt
COPY models models
COPY setup.py setup.py

RUN pip install -e .

# run container locally
CMD uvicorn heartbd.package_folder.api_file:app --reload —host 0.0.0.0

# run container deployed
# CMD uvicorn heartbd.package_folder.api_file:app --reload —host 0.0.0.0 —port $PORT
