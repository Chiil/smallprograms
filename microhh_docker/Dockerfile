# Use Ubuntu as the base
FROM ubuntu:16.04

# Add user and allow sudo
RUN useradd -ms /bin/bash microhh_user
RUN usermod -aG sudo microhh_user
COPY inputrc /home/microhh_user/.inputrc

# Complete install of required packages.
RUN apt-get update
RUN apt-get -y install \
        netcdf-bin libnetcdf-dev libnetcdf-c++4 libnetcdf-c++4-dev \
        fftw3 libfftw3-dev \
        git \
        cmake \
        python3 python3-pip \
        ncview \
        sudo \
        valgrind gdb binutils \
        vim
RUN pip3 install --upgrade pip
RUN pip3 install numpy matplotlib ipython jupyter netcdf4

# User actions.
USER microhh_user
WORKDIR /home/microhh_user
RUN git clone https://github.com/microhh/microhh
RUN git clone https://github.com/microhh/microhh2
WORKDIR microhh
RUN mkdir build
WORKDIR build
RUN cmake .. -DSYST=ubuntu
RUN make -j3
