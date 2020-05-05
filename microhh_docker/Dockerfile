# This Docker image can be run via the Docker cloud with the following command:
# docker run -it chiil/microhh bash

# The instructions below build the image from the Ubuntu 18.04 LTS image.

# Use Ubuntu as the base
FROM ubuntu:latest

# Complete install of required packages.
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
        netcdf-bin libnetcdf-dev libnetcdf-c++4 libnetcdf-c++4-dev \
        fftw3 libfftw3-dev \
        libboost-dev \
        git \
        cmake \
        python3 python3-pip python3-tk \
        ncview \
        sudo \
        valgrind gdb binutils \
        vim \
        libopenmpi-dev \
        gfortran
RUN pip3 install --upgrade pip
RUN pip3 install numpy matplotlib ipython jupyter netcdf4

# Add user and allow sudo
RUN useradd -p $(openssl passwd -1 microhh_user) -ms /bin/bash microhh_user
RUN usermod -aG sudo microhh_user

# User actions.
USER microhh_user
WORKDIR /home/microhh_user

# Suppress sudo message.
RUN touch .sudo_as_admin_successful

# Clone MicroHH source code.
RUN git clone https://github.com/microhh/microhh

# Echo instructions at login.
RUN echo "echo \"Open the MicroHH tutorial at: https://github.com/microhh/microhh/wiki/Tutorial\"" >> .bashrc
RUN echo "PS1='\u@\h:\w$ '" >> .bashrc
