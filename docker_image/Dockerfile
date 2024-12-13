FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

WORKDIR /root/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q && \
    apt-get install -y autotools-dev automake libtool python3-pip vim 

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
         git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
        net-tools \
  && rm -rf /var/lib/apt/lists/*

# install cmake
RUN apt-get update && \
    apt-get install -y build-essential wget libssl-dev && \
    wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0.tar.gz && \
    tar -xf cmake-3.22.0.tar.gz && \
    rm cmake-3.22.0.tar.gz && \
    cd cmake-3.22.0 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

# install python3.8
RUN apt-get update && apt-get install -y lzma liblzma-dev build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev && \
   wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz && tar -xf Python-3.8.16.tgz && \
   cd Python-3.8.16 && ./configure --enable-shared --prefix /usr --enable-optimizations && make -j$(nproc) && make altinstall
RUN ln -s /usr/lib/python3.8/_sysconfigdata__linux_x86_64-linux-gnu.py  /usr/lib/python3.8/_sysconfigdata__x86_64-linux-gnu.py

# install boost 1.76
RUN apt-get install -y g++ python-dev libicu-dev libboost-all-dev
RUN wget -c 'http://sourceforge.net/projects/boost/files/boost/1.76.0/boost_1_76_0.tar.bz2/download' && tar xf download && rm download
RUN cd boost_1_76_0 && ./bootstrap.sh --with-python=/usr/bin/python3.8
RUN cd boost_1_76_0 && ./b2 --prefix=/root/boost_1_76_0 variant=release link=shared threading=multi runtime-link=shared toolset=gcc install
ENV LD_LIBRARY_PATH=/root/boost_1_76_0/lib:$LD_LIBRARY_PATH

# install ompl for python3.8
RUN apt-get install -y mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev
# first, need to instal ODE
RUN git clone https://github.com/thomasmarsh/ODE.git
RUN cd ODE && mkdir -p build && cd build && cmake .. && make && make install

# second, need to install Py++
COPY ./download/castxml.tar.xz $HOME
RUN tar -xf $HOME/castxml.tar.xz -C $HOME && rm $HOME/castxml.tar.xz
ENV PATH=/root/castxml/bin:${PATH}

RUN apt-get update && \
      apt-get -y install sudo libeigen3-dev

RUN sudo -H pip3 install -vU pygccxml pyplusplus numpy

# then install ompl
RUN git clone https://github.com/ompl/ompl.git && cd ompl && mkdir -p build/Release && cd build/Release && \
    cmake ../.. -DPYTHON_EXEC=/usr/bin/python3 -DCASTXML=/root/castxml/bin/castxml && make -j$(nproc) update_bindings && make -j$(nproc) && make install

RUN sudo -H python3.8 -m pip install -vU pygccxml pyplusplus numpy

# ## Install yaml-cpp parser
# RUN git clone https://github.com/jbeder/yaml-cpp.git && cd yaml-cpp && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make && make install

# then install ompl for python3.8
RUN cd ompl && mkdir -p build_python3.8/Release && cd build_python3.8/Release && \
    cmake ../.. -DPYTHON_EXEC=/usr/bin/python3.8  -DCASTXML=/root/castxml/bin/castxml && make -j$(nproc) update_bindings && make -j$(nproc) && make install -j$(nproc)

# # uninstall yaml-cpp (not the original one)
# RUN cd yaml-cpp/build && make uninstall

# install libtorch
RUN apt-get install unzip && wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu117.zip && unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117.zip

# include CUDA to path
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH $CUDA_HOME/bin:$PATH

# # install jupyter
RUN python3 -m pip install --upgrade pip
RUN pip3 install --ignore-installed pyzmq && pip3 install jupyter

# Expose the port Jupyter will run on
EXPOSE 8888

# install open3d
RUN python3.8 -m pip install --ignore-installed scikit-learn PyYAML && python3.8 -m pip install open3d

# update skikit-image
RUN python3.8 -m pip install --upgrade scikit-image

# # install torch
RUN python3.8 -m pip install --upgrade pip && python3.8 -m pip install --upgrade pillow
RUN python3.8 -m pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3.8 -m pip install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install --no-cache-dir torch-geometric -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN python3.8 -m pip install einops
RUN python3.8 -m pip install tensorboard

# need to install opencv
RUN python3.8 -m pip install opencv-python open3d --ignore-installed PyYAML redis

# remove zip
RUN rm *.zip