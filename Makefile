
install_requirements:
	@pip install -r requirements.txt


install:
	@pip install -e .
	pip install --upgrade pip
#clean:
 #   @rm -f */version.txt
  #  @rm -f .coverage
   # @rm -f */.ipynb_checkpoints
    #@rm -Rf build
    #@rm -Rf */__pycache__
    #@rm -Rf */*.pyc
