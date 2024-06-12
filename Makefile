install_requirements:
	@pip install -r requirements.txt
	@echo "Make sure to run 'make install' before. Otherwise, if there is no error, proceed on your branch!"

install:
	@pip install -e .
	@pip install --upgrade pip

main_file_test:
	@python heartbd/interface/main.py

run_api:
	@uvicorn heartbd.package_folder.api_file:app --reload

# docker
dbuild_local:
	@open -a docker
	@echo "Opening Docker Desktop and logging in..."
	@sleep 10
	@docker login
	@docker build -t 'heartbeat_decoder' .

exports:
	@export IMAGE=heartbeat_decoder
	@export GCP_REGION=europe_west1
	@export ARTIFACTSREPO=heartbeat_decoder
	@export GCP_PROJECT=nifty-acolyte-424816-m3
	@export MEMORY=2Gi
	@export MODEL_TARGET='pickle'
	@export MODEL_PICKLE_PATH='heartbd/models/trained_model 2024-06-10-11H00.pkl'
