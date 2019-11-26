FROM nvcr.io/nvidia/pytorch:19.10-py3

ADD . /opt/project
WORKDIR /opt/project


RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt


