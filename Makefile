.PHONY: build
build:
	docker build -t bnn -f Dockerfile .

.PHONY: bash
bash:
	docker run -it --rm -p 8999:8888 -v ${PWD}:/code bnn bash

.PHONY: jupyter
jupyter:
	docker run -it --gpus all --rm -p 8999:8888 -v ${PWD}:/code bnn jupyter lab --allow-root --ip=*
