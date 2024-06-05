
#Pip commands

install_requirements:
	@pip install -r requirements.txt


install:
	@pip install -e .
	pip install --upgrade pip

# Tests commands

main_file_test:
	@python heartbd/interface/main.py


#clean:
 #   @rm -f */version.txt
  #  @rm -f .coverage
   # @rm -f */.ipynb_checkpoints
    #@rm -Rf build
    #@rm -Rf */__pycache__
    #@rm -Rf */*.pyc
