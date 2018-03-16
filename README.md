# A Pair-wise Learning to Rank Model using Random Forest


## Setup Environment
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
## Usage
For training:
```sh
$ python rf-ranker.py train path_to_train_data_file.tsv
```
For testing:
```sh
$ python rf-ranker.py test path_to_test_data_file.tsv"
```
## File schema
Files containing data points must have the following schema: Each instance must have two rows; a header row and a content row.

Example:

object_id \t data_point_id \t label \t feature_1 \t ..... \t feature_n

train_object_1 \t id_1 \t 0.4 \t 1.0 \t .... \t 0.5

For testing, "label" column is dropped.

## Code
The code is based on the following paper:
More Accurate Question Answering on Freebase [[pdf]](http://ad-publications.informatik.uni-freiburg.de/CIKM_freebase_qa_BH_2015.pdf)
Hannah Bast, Elmar Haussmann

You can find the original code at:
https://github.com/elmar_haussmann/aqqu

However, we re-implemented the code for easier accessibility.
