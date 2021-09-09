FROM pytorch/pytorch:latest

WORKDIR /code/
COPY requirements.txt /code/

RUN pip install -U pip
RUN pip install -r requirements.txt

ENV PYTHONPATH="/code/src"
COPY . /code/
