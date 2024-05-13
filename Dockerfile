FROM ubuntu:20.04
LABEL authors="junoh"
RUN apt-get update && \
    apt-get install -y \
      curl \
      bash \
      libgl1-mesa-glx \
      libegl1-mesa \
      libxrandr2 \
      libxrandr2 \
      libxss1 \
      libxcursor1 \
      libxcomposite1 \
      libasound2 \
      libxi6 \
      libxtst6 \
      ;
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
RUN bash Anaconda3-2024.02-1-Linux-x86_64.sh -b


ENV PATH /root/anaconda3/bin:$PATH

RUN conda init
ADD environment_others.yml environment.yml
RUN conda env create --file environment.yml
