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





# install_requirements:
# 	@pip install -r requirements.txt
# 	@echo "Make sure to run 'make install' before. Otherwise, if there is no error, proceed on your branch!"

# exports:
# 	@export IMAGE=heartbeat_decoder
# 	@export GCP_REGION=europe_west1
# 	@export ARTIFACTSREPO=heartbeat_decoder
# 	@export GCP_PROJECT=nifty-acolyte-424816-m3
# 	@export MEMORY=2Gi
# 	@export MODEL_TARGET='pickle'
# 	@export MODEL_PICKLE_PATH='heartbd/models/trained_model 2024-06-10-11H00.pkl'
# 	@echo "Completed .env exports!"

# install:
# 	@pip install -e .
# 	@pip install --upgrade pip

# main_file_test:
# 	@python heartbd/interface/main.py

# run_api:
# 	@uvicorn heartbd.package_folder.api_file:app --reload

# ################################# DOCKER #######################################
# dbuild_local:
# 	@docker build -t 'heartbeatdecoder' .

# drun_local:
# 	@docker run -it -p 8080:8000 'heartbeatdecoder'

# # first time only
# dallow:
# 	@gcloud auth configure-docker northamerica-northeast1-docker.pkg.dev

# cart_repo:
# 	@gcloud artifacts repositories create heartbeatdecoder --repository-format=docker \
# 	--location=northamerica-northeast1 --description="our cloud repository for hosting the model"

# # intel only
# dto_production:
# 	@docker build -t heartbeatdecoder:new .

# # push
# dpush:
# 	@docker push dcgale/heartbeatdecoder:new

# # deploy to cloud
# ddeploy:
# 	@gcloud run deploy --image dcgale/heartbeatdecoder:new --memory 2Gi --region northamerica-northeast1

# # northamerica-northeast1-docker.pkg.dev/nifty-acolyte-424816-m3/heartbeatdecoder:dev
# # northamerica-northeast1.pkg.dev/nifty-acolyte-424816-m3/heartbeatdecoder/heartbeatdecoder:0.1
