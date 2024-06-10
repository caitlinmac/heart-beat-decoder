
install_requirements:
	@pip install -r requirements.txt
	@echo "Make sure to run make install before otherwise if there is no error proceed on your branch."

install:
	@pip install -e .
	@pip install --upgrade pip

	
main_file_test:
	@python heartbd/interface/main.py
