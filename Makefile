# define the setup file
SETUP_FILE = setup.py
# define a flag file to indicate the package has been installed
PACKAGE_INSTALLED_FLAG = .package_installed
# define the requirements file
REQUIREMENTS_FILE = requirements.txt
# define a flag file to indicate the requirements are already installed
REQUIREMENTS_FLAG = .requirements_installed

# 'install_requirements' target to literally install the requirements
install_requirements: $(REQUIREMENTS_FLAG)
$(REQUIREMENTS_FLAG): $(REQUIREMENTS_FILE)
	@echo "Just checking to see if the requirements are satisfied..."
	@if pip freeze | grep -q -f $(REQUIREMENTS_FILE); then \
		echo "All requirements are already satisfied. Great!"; \
	else \
		echo "Installing requirements..."; \
		pip install -r $(REQUIREMENTS_FILE); \
		touch $(REQUIREMENTS_FLAG); \
		echo "Hooray! The requirements are now complete."; \
	fi

# 'install_package' target for installing 'heartbd' package
install_package: $(PACKAGE_INSTALLED_FLAG)
$(PACKAGE_INSTALLED_FLAG):$(SETUP_FILE)
	@echo "Just checking to see if you've already installed this as a package..."
	@if python -c heartbd 2>/dev/null; then \
    echo "Package is already installed."; \
	else  \
		echo "I guess not! Just installing the package for you..." \
		python $(SETUP_FILE) install; \
		touch $(PACKAGE_INSTALLED_FLAG); \
		echo "The package is now installed! Party time."; \
	fi

########################### DEFAULT AND CLEANUP ################################

# 'clean' target to remove the flag files and keep everything clean
clean:
	@rm -f $(REQUIREMENTS_FLAG) $(PACKAGE_INSTALLED_FLAG)
	@rm -fr */__pycache__
	@echo "We're all cleaned up now."

# set 'all' default target
all: install_requirements install_package
