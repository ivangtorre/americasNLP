FROM nvcr.io/nvidia/pytorch:22.05-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gedit \
    git \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-mp3 \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    nano \
    python3-pip \
    sox \
    wget \
    zlib1g \
    zlib1g-dev \
    liblzma-dev


RUN python -m pip install --upgrade pip

RUN python -m pip install \
    editdistance \
    pandas \
    sentencepiece \
    tensorboardX \
    tqdm



# INSTALL Flashlight Python Bindings
WORKDIR /opt
RUN git clone https://github.com/kpu/kenlm && cd kenlm && mkdir -p build && cd build && cmake .. && make -j 4
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

WORKDIR /opt
RUN git clone https://github.com/xianyi/OpenBLAS.git && cd OpenBLAS && mkdir -p build && cd build && \
    cmake .. && make -j 4 && make install

WORKDIR /opt
RUN wget http://www.fftw.org/fftw-3.3.10.tar.gz && tar -xvf fftw-3.3.10.tar.gz && cd fftw-3.3.10 && \
    mkdir build && cd build && cmake .. && make -j 4 && make install


WORKDIR /workspace
ENV KENLM_ROOT=/opt/kenlm
#ENV KENLM_LIB=/opt/kenlm/lib
#ENV KENLM_UTIL_LIB=/opt/util
#ENV KENLM_ROOT_DIR=/opt/kenlm/

COPY flashlight.zip .

RUN unzip flashlight && cd flashlight/bindings/python && \
    pip3 install -e .

#python3 setup.py install
RUN pip3 install hydra-ax-sweeper==1.2

## Install fairseq
COPY setup.py .
COPY audio_pretraining.py .
COPY infer.py .

RUN git clone https://github.com/pytorch/fairseq && cp setup.py fairseq/. && \
    cp audio_pretraining.py fairseq/fairseq/tasks/. && \
    cp infer.py fairseq/examples/speech_recognition/. && \
    cd fairseq && pip install --editable . && cp examples/speech_recognition/infer.py .
#    cd fairseq && python3 setup.py build_ext --inplace

WORKDIR /workspace