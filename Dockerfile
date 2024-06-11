FROM python:3.10.6

COPY heartbd heartbd
COPY requirements.txt
COPY models models

RUN pip install -e .

# Run container locally
CMD uvicorn heartbd/package_folder
