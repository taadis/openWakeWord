FROM python:3.7-slim
#FROM python:3.10-slim

RUN apt-get update && apt-get install --yes
#    git-lfs
#    && rm -rf /var/lib/apt/lists/*

WORKDIR /workdir
COPY . /workdir
#RUN pip3 install -r requirements.txt

# install full deps
RUN pip3 install -e "."
#RUN pip3 install -e ".[full]"
