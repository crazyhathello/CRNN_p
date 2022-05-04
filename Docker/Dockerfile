FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install --no-install-recommends -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN git clone https://github.com/crazyhathello/CRNN_p.git /CRNN_p
WORKDIR /CRNN_p

CMD ["python","./CRNN_test.py"]