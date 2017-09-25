FROM haoyangz/keras-docker:cuda8.0
MAINTAINER Haoyang Zeng  <haoyangz@mit.edu>

RUN pip install keras==1.2.0
RUN pip install --upgrade --no-deps git+https://github.com/maxpumperla/hyperas@0.2
RUN pip install --upgrade --no-deps git+https://github.com/hyperopt/hyperopt.git@0.0.2
RUN pip install pymongo==3.4.0 scikit-learn==0.18.1 networkx==1.11

ENV THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu0,floatX=float32,lib.cnmem=0.1,base_compiledir=/runtheano/.theano'

RUN git clone https://github.com/gifford-lab/keras-genomics.git /scripts/keras-genomics
RUN cd /scripts/keras-genomics;git fetch;git checkout keras1.2
RUN cp /scripts/keras-genomics/main.py /scripts/
RUN mkdir /runtheano/;chmod -R 777 /runtheano/
WORKDIR /scripts/
