FROM nvidia/cuda:10.0-cudnn7-devel

COPY . /opt/app/
RUN chmod -R 777 /opt/app
RUN echo 'root:root' | chpasswd
RUN apt-get update

RUN apt-get -y install openssh-server
RUN useradd -m nabatov
RUN chmod 600 /opt/app/keys/id_rsa

RUN apt-get -y install screen wget ssmtp lshw lsb-release sox software-properties-common libffi-dev libssl-dev libsox-fmt-mp3 

RUN cd /opt/app && \
     apt-get -y install software-properties-common && \
     add-apt-repository ppa:deadsnakes/ppa && \
     apt-get -y update && \
     apt-get -y install python3.6 && \
     apt-get -y install python3-dev && \
     apt-get -y install python3-pip && \
     apt-get -y install rsync grsync

RUN cd /opt/app && \
    python3.6 -m pip install --upgrade pip -r requirements_new.txt --src /usr/local/src 

RUN cp -r /opt/app/datasetimage/. /tmp/data/
RUN mkdir -p ~/.ssh
RUN touch ~/.ssh/authorized_keys
RUN cat /opt/app/keys/id_rsa.pub >> ~/.ssh/authorized_keys && chmod og-wx ~/.ssh/authorized_keys
RUN chown -R nabatov:nabatov ~/.ssh
RUN echo "Host remotehost\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
RUN /etc/init.d/ssh start
RUN service ssh start

EXPOSE 22

USER nabatov
#CMD python3.6 /opt/app/grid_search.py 
