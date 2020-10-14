# Source for Exploring Neural Entity Representations for Semantic Information

This repo contains the source code for the probing tasks and entity embeddings used in 
our paper "Exploring Neural Entity Representations for Semantic Information" (Runge and Hovy, 2020)
to be presented at BlackboxNLP 2020. 

## Download Pretrained Entity Embeddings

The pretrained entity embeddings for all eight models used in our paper are available
[here](https://drive.google.com/file/d/1Xm67sov7jxsKjFvEw9C79XgzKxbsklQ6/view?usp=sharing).
The entire compressed file is about 30GB.

Each embedding file is tab-delimited with the following structure:

```
entity_id0  embed_dim_0 embed_dim_1 embed_dim_2 ...
entity_id1  embed_dim_0 embed_dim_1 embed_dim_2 ...
...
```

Each `entity_id` corresponds to the MediaWiki ID # for the corresponding Wikipedia page. 

## Probing Tasks

The probing task datasets are in the `probing_tasks` directory of this repository.
Each task directory contains one or more subtasks, which in turn each have a `train.txt`
and `test.txt` file. 

There are two main formats for these files, depending on the task: 

* A single-entity task (either classification or regression), with the format

```
entity_id0  label0
entity_id1  label1
...
``` 

* A paired-entity task with the format

```
entity_id0a entity_id0b label0
entity_id1a entity_id1b label1
...   
```

## Creating the Probing Datasets

The probing datasets can be constructed using the embeddings and probing task
files using `src/train_generator.py`. The script can be run as follows:

```
train_generator.py [-h] -e EMBEDS [-t TASKS] [-o OUTPUT] [-m MODELS]

optional arguments:
  -h, --help            show this help message and exit
  -e EMBEDS, --embeds EMBEDS
                        Directory containing embedding files
  -t TASKS, --tasks TASKS
                        Directory containing task datasets
  -o OUTPUT, --output OUTPUT
                        Directory where probing datasets will be written
  -m MODELS, --models MODELS
                        Comma-separated list of which embedding model(s)
                        should be used to create probing datasets. Default:
                        All - creates a dataset for each of the 8 models used
                        in the paper.
```

After downloading the embeddings from [here](https://drive.google.com/file/d/1Xm67sov7jxsKjFvEw9C79XgzKxbsklQ6/view?usp=sharing),
unzip them into the `embeds` directory in the root of the repository. The probing
datasets can then be constructed for all embedding models using the following command:

`python src/train_generatory.py -e embeds/ -t tasks/ -o probing-datasets/`

Constructing the probing datasets for all models requires 63.8GB of free space. You
can instead only create the datasets for a single model by providing a comma-separated
list of model names (model names are the same as their subdirectory in the embeddings
file). For example:

`python train_generatory.py -e embeds/ -t tasks/ -o probing-datasets/ -m "wiki2vec,ganea,bert-base"`

### Running the Probing Tasks

Once the datasets are built, you can run the probing tasks using `src/probing.py`:

```
usage: probing.py [-h] [-d DATASETS] [-t TASKS] [-o OUTPUT] [-m MODELS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASETS, --datasets DATASETS
                        Directory containing probing task datasets. Default: ../probing-datasets/
  -t TASKS, --tasks TASKS
                        Comma-separated list of tasks to run. Default: All
  -o OUTPUT, --output OUTPUT
                        Directory where results of probing experiments will be
                        written. Default: ../experiments/
  -m MODELS, --models MODELS
                        Comma-separated list of which embedding datasets will
                        be evaluated. Default: All

```

For example, to run only all of the context words tasks for the CNN and RNN models, 
run the following inside the `src` directory:

`python probing.py -t context_words_high_freq,context_words_mid_freq entity_types -m cnn`

The results will be written to the specified output directory in two files per task category.
The first is `<task_name>-exp-results.txt`, which reports task F1 and/or RMSE for each of the
subtasks. The second is `<task_name>-experiment-full-results.txt` which reports confusion
matrices for each subtask in addition to core performance metrics.
 