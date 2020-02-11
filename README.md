# CodeNets

> My own playground to play with PLP (Programming Language Processing) datasets & Deep Learning

In this repository, I want to study **_PLP_, programming languages processing** (searching, modifying, generating, translating my code...) using AI techniques like Deep Learning.

At the same time, I want to use this code base to evaluate advanced programming techniques with Python (yes it is possible to do that), AI Libraries (Pytorch for now but could use others later) using typesafe programming (Mypy) and some functional programming techniques with becoming too mad.

## Current Studies

### [CodeSearchNet](https://github.com/github/CodeSearchNet)

The current code is a ~80% rewrite of the original [Github repository](https://github.com/github/CodeSearchNet) open-sourced by Microsoft team with a paper and blog post with a benchmark on W&B and the Github CodeSearchNet dataset.

> Why did I rewrite an already complete library?
- to use the dataset independently of Tensorflow as existing code is really focused on it (Pytorch is my current choice for NLP until I find another one ;)).
- to be able to use the best NLP Deep Learning library for now [Huggingface Transformers](https://github.com/huggingface/transformers) and more recently their new beta project [Huggingface Rust Tokenizers](https://github.com/huggingface/tokenizers) and I must say it's much better used in Pytorch.

> What does it provide compared to original repo?

- Dataset loading & parsing is independent from Tensorflow and memory optimized (original repo couldn't fit in my 32GB CPU RAM)
- Support of Pytorch,
- Support of Huggingface transformers pretrained & non-pretrained ([Sample of Bert from scratch](./codenets/codesearchnet/multi_branch_model.py#L103-L123))
- Support of Huggingface Rust tokenizers and training them ([Sample of tokenizer training](./codenets/codesearchnet/tokenizer_recs.py#L425-L452)),
- Mostly typed Python (with Mypy) ([sample code](./codenets/codesearchnet/multi_branch_model.py)),
- experimental typesafe "typeclass-like helpers" to save/load full Training heterogenous contexts (models, optimizers, tokenizers, configuration using different libraries) ([a sample recordable Pytorch model](./codenets/codesearchnet/multi_branch_model.py#L31-L64) and [a full recordable training context](https://github.com/mandubian/codenets/blob/master/codenets/codesearchnet/multi_branch_model.py#L163-L265))
- HOCON configuration for full models and trainings ([sample config](./conf/default.conf)),
- Poetry Python dependencies management with isolated virtualenv.



## Next potential Studies

- Deep Learning bug correction by AST modification
- Function generation from function signature
- Type error detection by Deep Learning
- Type logic & reasoning by Deep Learning


## Installing project

### Install poetry

Following instructions to install [Poetry](https://python-poetry.org/docs/).

>Why poetry instead of basic requirements.txt?

Because its dependency management is more automatic. Poetry has big defaults & bugs but its dependency management is much more production ready than other Python solutions I've tested and it isolates by default your python environment in a virtual env (The other best solution I've found is a hand-made virtualenv in which I install all my dependencies with a requirements.txt).

### Install a decent Python (with pyenv for example)

```sh
pyenv local 3.7.2
```

> Please python devs, stop using Python 2.x, this is not possible anymore to use such bloated oldies.

### Install & Build Virtual Env

```sh
poetry install
poetry shell
```

Now you should be in a console with your virtualenv environment and all your custom python dependencies. Now you can run python.

## Training [CodeSearchNet](https://github.com/github/CodeSearchNet) models

### Training model

1. Write a HOCON configuration for your training

- Copy a configuration file from `conf` directory and modify it according to your needs.
- Take care to give a `training.name` to your experiment and a unique `training.iteration` to your current run.

- Choose a training context class

```
# Query1Code1: Single Bert query encoder, Single Bert encoders
training_ctx_class = "codenets.codesearchnet.query_1_code_1.training_ctx.Query1Code1Ctx"

# Query1CodeN: Single Bert query encoder, Multiple Bert encoders
training_ctx_class = "codenets.codesearchnet.query_1_code_1.training_ctx.Query1CodeNCtx"
```

2. Train Tokenizers in this training context

```sh
python codenets/codesearchnet/tokenizer_build.py --config ./conf/my_conf_file.conf
```
3. Train Model

```sh
python codenets/codesearchnet/train.py --config ./conf/my_conf_file.conf
```

4. Eval Model

```sh
python codenets/codesearchnet/eval.py --restore ./checkpoints/YOUR_RUN_DIRECTORY
```

5. Build Benchmark predictions

```sh
python codenets/codesearchnet/predictions.py --restore ./checkpoints/YOUR_RUN_DIRECTORY
```


### Models

#### Query1CodeN: Single Bert query encoder, Multiple Bert encoders (one encoder per language)

_Incoming_

#### Query1Code1: Single Bert query encoder, Single Bert encoder for all languages

_Incoming_


### HOCON configuration

_Incoming_

## For future, pre-Trained models ?

I'll publish my models if I can reach some good performances (this is not the case yet)
