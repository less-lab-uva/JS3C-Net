FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

# https://gitlab.com/nvidia/container-images/cuda/-/issues/158
RUN apt-key del "7fa2af80" \
&& export this_distro="$(cat /etc/os-release | grep '^ID=' | awk -F'=' '{print $2}')" \
&& export this_version="$(cat /etc/os-release | grep '^VERSION_ID=' | awk -F'=' '{print $2}' | sed 's/[^0-9]*//g')" \
&& apt-key adv --fetch-keys "http://developer.download.nvidia.com/compute/cuda/repos/${this_distro}${this_version}/x86_64/3bf863cc.pub" \
&& apt-key adv --fetch-keys "http://developer.download.nvidia.com/compute/machine-learning/repos/${this_distro}${this_version}/x86_64/7fa2af80.pub"


RUN apt update && apt upgrade -y
RUN apt install -y git wget unzip
RUN apt-get install -y ninja-build

# https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in
# RUN apt-get install -y build-essential
RUN apt install build-essential software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt update
RUN apt install gcc-6 g++-6 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6 && gcc -v

RUN apt install -y libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev

# https://moreless.medium.com/install-python-3-6-on-ubuntu-16-04-28791d5c2167
RUN apt-get install -y zlib1g-dev
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
RUN tar -xvf Python-3.6.9.tgz
WORKDIR /opt/Python-3.6.9
RUN ./configure
RUN make 
RUN make install
WORKDIR /

# get newer version of cmake
RUN wget https://apt.kitware.com/kitware-archive.sh
RUN chmod +x kitware-archive.sh && ./kitware-archive.sh

RUN pip3 install numpy==1.17.3 cython==0.29.14 tqdm==4.37.0 pyyaml==5.1.1 Pillow==6.2.1 torch==1.3.1+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt install -y libjpeg-dev zlib1g-dev


ENV PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV CUDA_HOME=/usr/local/cuda-10.1
ENV CUDA_PATH=/usr/local/cuda-10.1
ENV FORCE_CUDA="1"
# RUN TORCH_CUDA_ARCH_LIST="6.1"
RUN python3 -c "import torch;print(torch.cuda.is_available())" | grep 'True'

# https://github.com/facebookresearch/SparseConvNet/issues/96
RUN apt-get install -y libsparsehash-dev

RUN ls
RUN git clone https://github.com/GarrettChristian/JS3C-Net.git
WORKDIR /JS3C-Net/lib/
RUN python3 setup.py develop
WORKDIR /JS3C-Net/lib/nearest_neighbors
RUN python3 setup.py install
WORKDIR /JS3C-Net/lib/pointgroup_ops
RUN python3 setup.py develop
WORKDIR /



# SpConv
# need to use SpConv 1
RUN git clone --recursive https://github.com/traveller59/spconv.git /spconv
WORKDIR /spconv
RUN git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634 
RUN git submodule update --init --recursive

RUN apt install -y libboost-all-dev cmake
ENV CUDA_ROOT=/usr/local/cuda
RUN pip3 install wheel
RUN python3 setup.py bdist_wheel
WORKDIR /spconv/dist
RUN pip3 install *.whl
WORKDIR /