install_requirements:
	@pip install -r requirements.txt
	@echo "Make sure to run 'make install' before. Otherwise, if there is no error, proceed on your branch!"

install:
	@pip install -e .
	@pip install --upgrade pip

main_file_test:
	@python heartbd/interface/main.py

# API
run_api_local:
	@uvicorn heartbd.package_folder.api_file:app --reload

# docker
build_docker:
	@docker build -t $$ARTIFACTSREPO .

run_docker:
	@docker run -it -e PORT=0000 -p 8000:8000 'heartbeat_decoder'

# Step 1 (First time only)
allow_docker_push:
	@gcloud auth configure-docker $$GCP_REGION-docker.pkg.dev

# Step 2 (First time only)
create_artifacts_repo:
	@gcloud artifacts repositories create $$ARTIFACTSREPO --repository-format=docker --location=$$GCP_REGION --description="Repositiory of heart beat decoder"

# Step 3
build_for_production:
	@docker build -t $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE .

# Step 4
push_image_production:
	@docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE

# Step 5
deploy_and_run:
	@gcloud run deploy --image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE

exports:
	@export IMAGE=heartbeat_decoder
	@export GCP_REGION=europe_west1
	@export ARTIFACTSREPO=heartbeat_decoder
	@export GCP_PROJECT=nifty-acolyte-424816-m3
	@export MEMORY=2Gi
	@export MODEL_TARGET='pickle'
	@export MODEL_PICKLE_PATH='heartbd/models/trained_model 2024-06-10-11H00.pkl'
