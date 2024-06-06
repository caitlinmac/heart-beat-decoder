.PHONY: install_requirements

install_requirements:
	@pip install -r requirements.txt
	@echo "Hooray, the requirements are complete!"

install:
	@pip install -e .
	pip install --upgrade pip
