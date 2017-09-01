# Use Ubuntu as the base
FROM ubuntu:16.04

# Complete install of required packages.
RUN apt-get update
RUN apt-get -y install \
        netcdf-bin libnetcdf-dev libnetcdf-c++4 libnetcdf-c++4-dev \
        fftw3 libfftw3-dev \
        git \
        cmake \
        python3 python3-pip python3-tk \
        ncview \
        sudo \
        valgrind gdb binutils \
        vim
RUN pip3 install --upgrade pip
RUN pip3 install numpy matplotlib ipython jupyter netcdf4

# Add user and allow sudo
RUN useradd -p $(openssl passwd -1 microhh_user) -ms /bin/bash microhh_user
RUN usermod -aG sudo microhh_user

# User actions.
USER microhh_user
WORKDIR /home/microhh_user
RUN git clone https://github.com/microhh/microhh
WORKDIR microhh
RUN mkdir build
WORKDIR build
RUN cmake .. -DSYST=ubuntu
RUN make -j3
