# A Learning to Rank Model using Random Forest


## Setup Environment:
First you need to install dependencies as follows:
- Create a virtualenv:
```sh
$ virtualenv venv
$ source venv/bin/activate
```
- Run:
```sh
$ pip install -r requirements.txt
```
## Usage:
For training:
```sh
$ python rf-ranker.py train path_to_train_data_file.tsv
```
For testing:
```sh
$ python rf-ranker.py test path_to_test_data_file.tsv"
```

The code is based on the following paper:
More Accurate Question Answering on Freebase [[pdf]](http://ad-publications.informatik.uni-freiburg.de/CIKM_freebase_qa_BH_2015.pdf)
Hannah Bast, Elmar Haussmann

You can find the original code at:
https://github.com/elmar_haussmann/aqqu

However, we re-implemented the code for easier accessibility.
