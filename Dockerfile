FROM pytorch/pytorch:latest

RUN pip install -U pip
RUN pip install jupyterlab matplotlib sklearn tensorboard ipywidgets
WORKDIR /code/  
