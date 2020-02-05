# CodeNets

> My own playground to play with PLP (Programming Language Processing) datasets & Deep Learning

In this repository, I want to study **_PLP_, programming languages processing**(searching, modifying, generating, translating...) using AI techniques like Deep Learning.

At the same time, I want to use this code base to evaluate advanced programming techniques with Python, AI Libraries (Pytorch for now but could use others later) and typed programming (Mypy) and some functional programming techniques.

## Current Studies

### [CodeSearchNet](https://github.com/github/CodeSearchNet)

This is a ~80% rewrite of the original [Github repository](https://github.com/github/CodeSearchNet) open-sourced by Microsoft team based on a paper and blog post. This repository is quite good and complete but is exclusively focused on Tensorflow even for the dataset part. It also manages a benchmark on WanDB for the CodeSearchNet dataset.

> Why did I rewrite an already complete library?

I want to be able:

- to use the dataset indenpendently of Tensorflow which is not the library I prefer (Pytorch is my current candidate).
- to be able to re-use the best NLP Deep Learning library for now [Huggingface Transformers](https://github.com/huggingface/transformers) and more recently their new beta project [Huggingface Tokenizers](https://github.com/huggingface/tokenizers) and I must say it's much better used in Pytorch.

> What does it provide compared to original repo?

- Dataset loading & parsing is independent from Tensorflow and memory optimized (original repo couldn't fit in my 32GB CPU RAM)
- Support of Pytorch,
- Samples of models and trainings with HuggingFace transformers & tokenizers,
- Mostly typed Python (with Mypy) ([sample code](./codenets/codesearchnet/multi_branch_model.py#L31-L64)),
- some typesafe experimental "typeclass-like helpers" to save/load full Training heterogenous contexts (models, optimizers, tokenizers, configuration using different libraries) ([sample code](./codenets/blob/master/codenets/codesearchnet/multi_branch_model.py#L31-L64))
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

#### Single Bert query encoder, Multiple Bert encoders (one encoder per language)

_Incoming_

#### Single Bert query encoder, Single Bert encoder for all languages

_Incoming_

### Evaluating model

_Incoming_

### Infering model

_Incoming_

## For future, pre-Trained models ?

I'll publish my models if I can reach some good performances (this is not the case yet)
