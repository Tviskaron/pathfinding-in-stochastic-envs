FROM  pytorch/pytorch
RUN apt update && apt install build-essential ffmpeg libsm6 libxext6 git vim -y
ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

