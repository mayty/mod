# Versions
PYTHON = python3.9.7
PIP_VERSION = 21.2.4
SETUP_TOOLS_VERSION = 57.0.4


# Paths
PROJECT_PATH = $(CURDIR)
PIP_DEV_CONFIG=pip/dev.conf

all: help
$(VENV_ACTIVATE):
	PIP_CONFIG_FILE=$(PIP_DEV_CONFIG) pip install pip==$(PIP_VERSION) setuptools==$(SETUP_TOOLS_VERSION)

.PHONY: typecheck
typecheck: $(VENV_ACTIVATE)
	mypy -p mod

check: typecheck

.PHONY: install
install: $(VENV_ACTIVATE)
	pip install --upgrade -r requirements.txt

