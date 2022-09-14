# Pull the latest image of CentOS from the Docker repo
FROM centos:7

# Creator
MAINTAINER Karl W. Schulz

# Define a user
RUN useradd -u 2000 -m ohpc
RUN useradd -u 1000 -m karl

# Disable yum fastmirror checks
RUN sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf

# Add some packages
RUN yum -y install epel-release
RUN yum -y install python36 python36-devel
RUN yum -y install python36-pip
RUN yum -y install libgomp
RUN yum -y install gcc
RUN yum -y install gcc-c++

# Include python3 packages from pip
COPY requirements.txt /
RUN pip3.6 install -r requirements.txt

# User to run as
USER karl
WORKDIR /home/karl/
CMD jupyter notebook --ip 0.0.0.0 --no-browser
