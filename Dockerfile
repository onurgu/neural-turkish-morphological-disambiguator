FROM ubuntu:16.04

RUN mkdir /opt/neural-turkish-disambiguator
WORKDIR /opt/neural-turkish-disambiguator/

RUN apt-get update -y && apt-get install python-pip python-dev -y

RUN pip install --upgrade pip
RUN pip install wheel

RUN pip install --upgrade https://pypi.python.org/packages/7b/c5/a97ed48fcc878e36bb05a3ea700c077360853c0994473a8f6b0ab4c2ddd2/tensorflow-1.0.0-cp27-cp27mu-manylinux1_x86_64.whl#md5=a7483a4da4d70cc628e9e207238f77c0

COPY scripts /opt/neural-turkish-disambiguator/

COPY public_html/ /opt/neural-turkish-disambiguator/public_html/

COPY tools/tr-tagger /opt/neural-turkish-disambiguator/tools/tr-tagger/

COPY requirements.txt /opt/neural-turkish-disambiguator/
RUN pip install -r requirements.txt

COPY *.py /opt/neural-turkish-disambiguator/

RUN mkdir /opt/neural-turkish-disambiguator/models
COPY models/ntd-nmd-20170619-06.epoch-32-val_acc-0.99507.hdf5 /opt/neural-turkish-disambiguator/models
COPY models/ntd-nmd-20170619-06.epoch-32-val_acc-0.99507.hdf5.label2ids /opt/neural-turkish-disambiguator/models

EXPOSE 10001

# CMD python webapp.py --command disambiguate --train_filepath data/train.merge.utf8 --test_filepath data/test.merge.utf8 --model_path ./models/ntd-nmd-20170619-06.epoch-32-val_acc-0.99507.hdf5 --label2ids_path ./models/ntd-nmd-20170619-06.epoch-32-val_acc-0.99507.hdf5.label2ids --run_name testing --port 10001
CMD bash