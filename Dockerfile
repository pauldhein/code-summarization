# use a container for python 3
FROM python:3

# to install dependencies with pip, see the following example
RUN pip install nltk==3.3
RUN pip install numpy==1.14.5
RUN pip install dynet

RUN mkdir /workspace
RUN mkdir /workspace/data
RUN mkdir /workspace/utils
RUN mkdir /workspace/vector-maker

WORKDIR /workspace

COPY data /workspace/data
COPY utils /workspace/utils
COPY vector-maker /workspace/vector-maker

ADD generate.py /workspace
ADD bleu.py /workspace
ADD score_models.sh /workspace

CMD [ "./score_models.sh" ]
