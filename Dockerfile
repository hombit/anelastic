FROM continuumio/miniconda3

MAINTAINER Konstantin Malanchev <kmalanchev@hse.ru>

VOLUME /dumps
VOLUME /results

ENV PROJECT /anelastic
RUN mkdir -p $PROJECT
WORKDIR $PROJECT

COPY requirements.txt $PROJECT/
RUN conda install --yes --file requirements.txt &&\
    conda install --yes ipython &&\
    conda clean --yes --all

COPY . $PROJECT
RUN python setup.py install

CMD ["modal_analysis", "--dumpspath=/dumps", "--resultspath=/results"]
