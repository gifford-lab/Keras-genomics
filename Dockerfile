FROM haoyangz/keras-docker
MAINTAINER Haoyang Zeng  <haoyangz@mit.edu>

RUN pip install --upgrade --no-deps git+https://github.com/maxpumperla/hyperas@0.2
RUN pip install --upgrade --no-deps hyperopt pymongo scikit-learn networkx

ENV THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu0,floatX=float32,lib.cnmem=0.1,base_compiledir=/runtheano/.theano'

COPY main.py /scripts/
RUN mkdir /runtheano/;chmod -R 777 /runtheano/
WORKDIR /scripts/
