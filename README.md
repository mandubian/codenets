# CodeNets

**A playground to play with PLP (Programming Language Processing) datasets & Deep Learning**

> Code is currently under heavy work so it's not stable but all the principles are there.

In this repository, I want to study **_PLP_, programming languages processing** (searching, modifying, generating, translating my code...) using AI techniques like Deep Learning.

At the same time, I want to use this code base to evaluate advanced programming techniques with Python (yes it is possible to do that), AI Libraries (Pytorch for now but could use others later) using typesafe programming (Mypy) and some functional programming techniques with becoming too mad.

**If you have GPU(s) to share with me for some time to perform more experiments on those topics, don't hesitate to contact me here on github or on twitter [@mandubian](http://twitter.com/mandubian).**

## Current Studies

### [CodeSearchNet](https://github.com/github/CodeSearchNet)

The current code is a ~80% rewrite of the original [Github repository](https://github.com/github/CodeSearchNet) open-sourced by Microsoft team with a paper and blog post with a benchmark on W&B and the Github CodeSearchNet dataset.

> Why did I rewrite an already complete library?
- to use the dataset independently of Tensorflow as existing code is really focused on it (Pytorch is my current choice for NLP until I find another one ;)).
- to be able to use the best NLP Deep Learning library for now [Huggingface Transformers](https://github.com/huggingface/transformers) and more recently their new beta project [Huggingface Rust Tokenizers](https://github.com/huggingface/tokenizers) and I must say it's much better used in Pytorch.

> What does it provide compared to original repo?

- Dataset loading & parsing is independent from Tensorflow and memory optimized (original repo couldn't fit in my 32GB CPU RAM)
- Support of Pytorch,
- Support of Huggingface transformers pretrained & non-pretrained: [Sample of Bert from scratch](./codenets/codesearchnet/query_1_code_1/model.py#L111-L126)
- Support of Huggingface Rust tokenizers and training them: [Sample of tokenizer training](./codenets/codesearchnet/query_1_code_1/training_ctx.py#L226-L250),
- Mostly typed Python (with Mypy) ([sample code](./codenets/recordable.py#L13-L25)),
- experimental typesafe "typeclass-like helpers" to save/load full Training heterogenous contexts (models, optimizers, tokenizers, configuration using different libraries): [a sample recordable Pytorch model](./codenets/codesearchnet/query_1_code_1/model.py#L33-L66) and [a full recordable training context](./codenets/codesearchnet/query_1_code_1/model.py#L33-L66))
- HOCON configuration for full models and trainings: [sample config](./conf/default.conf)),
- Poetry Python dependencies management with isolated virtualenv ([Poetry config](./pyproject.toml).

## Trainings

I'm currently experimenting different models & runs are recorded on W&B

https://app.wandb.ai/mandubian/codenets

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

#### Write your HOCON configuration for your training

- Copy a configuration file from `conf` directory and modify it according to your needs.
- Take care to give a `training.name` to your experiment and a unique `training.iteration` to your current run. You can have several `training.iteration` for one `training.name`.

- Choose a training context class

```
# Query1Code1: Single Bert query encoder+tokenizer, Single Bert encoder+tokenizer for all coding languages
training_ctx_class = "codenets.codesearchnet.query_1_code_1.training_ctx.Query1Code1Ctx"

# Query1CodeN: Single Bert query encoder+tokenizer, Single encoder+tokenizer per language
training_ctx_class = "codenets.codesearchnet.query_1_code_1.training_ctx.Query1CodeNCtx"

# QueryCodeSiamese: Single Bert encoder+tokenizer for query and coding languages
training_ctx_class = "codenets.codesearchnet.query_code_siamese.training_ctx.QueryCodeSiamese"
```

#### Train Tokenizers in this training context

```sh
python codenets/codesearchnet/tokenizer_build.py --config ./conf/MY_CONF_FILE.conf
```

#### Train Model

```sh
python codenets/codesearchnet/train.py --config ./conf/MY_CONF_FILE.conf
```

> It should store your pickles in `training.pickle_path` and your checkpoints in `training.checkpoint_path`.

#### Eval Model

```sh
python codenets/codesearchnet/eval.py --restore ./checkpoints/YOUR_RUN_DIRECTORY
```

#### Build Benchmark predictions

```sh
python codenets/codesearchnet/predictions.py --restore ./checkpoints/YOUR_RUN_DIRECTORY
```


### Experiments

> I haven't submitted any model to official leaderboard https://app.wandb.ai/github/codesearchnet/benchmark/leaderboard because you can submit only every 2 weeks and I'm experimenting many different models during such long period so I prefer to keep searching for better models instead of submitting. But you can compute your NDCG metrics on the benchmark website and it is stored in your W&B run so this is what I use for now to evaluate my experiments.


_Please note that all experiments are currently done on my own 1080TI GPU on a 32GB workstation. So being strongly limited by my resources and time, I have to choose carefully my experiments and can't let them run forever when they do not give satisfying results after a few epochs._

#### Query1CodeN: Single Bert Query encoder/tokenizer, N Bert code encoders/tokenizers

- 1 x (query tokenizer + query encoder)
- N languages x (code tokenizer + code encoder)

##### Experiments

This is the model from original paper https://arxiv.org/pdf/1909.09436.pdf and I haven't pushed such trainings too far as the authors did it. You can check whole results in the paper.

Just for information here are their MRR and NDCG results for:
- NBOW a simple NN with a linear token embedding (providing the baseline)
- Bert-like encoding with self-attention

Both are trained using a BPE-style Vocabulary of size 10.000
       
|`MRR`| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| NBOW |**0.6167**|0.6409|0.5140|0.4607|0.4835|0.5809|0.4285|
| Bert-Like |**0.7011**|0.6809|0.5866|0.4506|0.6011|0.6922|0.3651|

|`NDCG`| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| NBOW |**0.164**|0.130|0.121|0.175| 0.123|0.223|0.212|
| Bert Like |**0.113**|0.049|0.100|0.061|0.094|0.181|0.190|

Interestingly, Bert-like models reach much better MRR on training compared to NBOW but not on NDCG. MRR doesn't care about the relevance of the result, just the rank. NDCG cares about relevance and rank. But when you train using default data, there is no relevance, the choice is binary.

In terms of languages, NBOW is better in all cases and behaves better on Java/Python/Ruby than Go/JS/PHP. Bert-like models are also better for Python/Java/Ruby but very bad on PHP, Go and worse in Javascript. Python/Java/Ruby are fairly simple, sequentially structured and imperative languages so maybe seq-aware models can rely on that. For Javascript, when you look at code samples, you see that JS is a language with lots of anonymous functions/callbacks so the code is much less sequentially interpretable. PHP is quite messy language to me but I would need a deeper analysis to know why it's harder. For Go, I cannot really give any good interpretation without deeper qualitative study.

But it's a bit surprising that simpler NBOW model gives far better result than BERT.

----

#### Query1Code1: Single Bert query encoder, common Bert encoder for all languages

This model is my first target experiment: N encoders is not viable as I am a poor data scientist in my free-time and have only one GPU and 32GB of RAM.

So, I decided to try:
- **1 query encoder+tokenizer (Bert Encoder + BPE Tokenizer)**
- **1 common code encoder+tokenizer for all languages (Bert Encoder + BPE Tokenizer)**.


> Bert models are trained from scratch using well-know [Huggingface transformers library](https://huggingface.co/).
>
> BPE Tokenizers are trained from scratch using new [Hugginface Rust Tokenizer library](https://github.com/huggingface/tokenizers).

##### Experiment 2020/02/10

###### Configuration

BPE Vocabulary size is set to 10K for query and code tokenizers.

Output embedding size is set to 128.
Query Encoder is a smaller BERT than code encoder to reflect the more complex 5 languages in one single BERT.

|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query Path|Bert 512/3/8/128|BPE 10K|
|1 Code Path|Bert 1024/6/8/128|BPE 10K|

|`Training`|epochs|lr|loss|batch size|seed|epoch duration|
|---|---|---|---|---|---|---|
||10|0.0001|softmax_cross_entropy|170|0|~1h30|

###### W&B Run

https://app.wandb.ai/mandubian/codenets/runs/j12z3vfr/overview?workspace=user-mandubian

###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.9536|0.8551|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.1014**|0.1294|0.1027|0.0407|0.07784|0.1364|0.1212|

###### Analysis

We see that MRR is much higher than what was obtained in CodeSearchNet paper (0.95 vs 0.70) but the NDCG is lower (0.10 vs 0.11).

`Query1Code1` has learnt to find the right exact piece of code from a query and it seems quite good at it (despite we haven't the MRR on benchmark dataset). But in terms of ranking pieces of code between each other with a relevance ranking (which is what NDCG represents), it's not good. We haven't the MRR on benchmark dataset so it's hard to know if our model tends to overfit on codesearchnet dataset (due to an unknown bias) and if it's a generalization issue.

Yet, this also demonstrates that MRR and NDCG are 2 different metrics and evaluating your model on MRR is not really enough when the benchmark metrics is NDCG. NDCG takes into account the relevance of pieces of code between each others and the ranking. In the dataset, we haven't relevance of piece of codes between each other so it's not trivial to train on a ranking+relevance objective.

----

#### QueryCodeSiamese: common Bert encoder+tokenizer for query and all languages.

Having several BERT models on my 11GB 1080TI limits a lot the size of each model and batches. Thus, I can't train deeper model able to store more knowledge (recent studies show that bigger NLP models store more "knowledge"). Moreover, not being able to increase batch size reduces the speed of training too.

So I've decided to try the most extreme case: **one single shared BERT and tokenizer for query and tokenizer** and see what happens.


The model is then:
- **1 encoder+tokenizer (Bert Encoder + BPE Tokenizer) for query + all languages**


> Bert models are trained from scratch using well-know [Huggingface transformers library](https://huggingface.co/).
>
> BPE Tokenizers are trained from scratch using new [Hugginface Rust Tokenizer library](https://github.com/huggingface/tokenizers).


##### Experiment 2020/02/14

###### Configuration

BPE Vocabulary size is set to 60K which is 6 times bigger than previous experiment to reflect that it must encode both query (which are almost normal sentences) and code from 5 languages (with lots of technical tokens like `{}[]...`, acronyms and numbers). No further token techniques like snake/camel-case subtokenization have been applied till now.

In this 1st experiment on this model, output embedding size is set to smaller 72 (<128) to test capability of model to learn a lot with less encoding space. But then I've fixed the intermediate BERT size also to a smaller 256 size while increasing the number of heads and layers to 12 to give the model more capabilities in terms of diversity and depth.

|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query+Code Path|Bert 256/12/12/72|BPE 60K|

|`Training`|epochs|lr|loss|batch size|seed|epoch duration|
|---|---|---|---|---|---|---|
||10|0.0001|softmax_cross_entropy|100|0|~3h|

###### W&B Run

https://app.wandb.ai/mandubian/codenets/runs/f6ebrliy/overview?workspace=user-mandubian

###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.9669|0.8612|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.1375**|0.1021|0.1475|0.04765|0.1068|0.2039|0.217|


###### Analysis

So with a single encoder/tokenizer of smaller size and smaller batches, we see that MRR is slightly higher than `Query1Code1`. The NDCG is also a bit higher (0.13 vs 0.10) but lower than NBOW baseline. The performance on the languages remain consistent (better on java/python/ruby and not on go/js/ruby and very bad in js).

With a shared model and tokenizer with smaller embedding and intermediate size but deeper layers and heads, we haven't lost any performance compared to separated branches. But we haven't reached the baseline performance.

Now let's see what happens with same siamese configuration but an even smaller embedding size and model.


##### Experiment 2020/02/17

###### Configuration

Same configuration as before but with smaller output embedding size of 64 and smaller BERT model but a bigger batch size of 290 which accelerates the training.

|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query+COde Path|Bert 256/6/8/64|BPE 60K|
----
|`Training`|epochs|lr|loss|batch size|seed|epoch duration|
|---|---|---|---|---|---|---|
||10|0.0001|softmax_cross_entropy|290|0|~1h10|

###### W&B Run

https://app.wandb.ai/mandubian/codenets/runs/ath9asmp/overview?workspace=user-mandubian

###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.9286|0.7446|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.1707**|0.1527|0.1716|0.07945|0.1437|0.2492|0.2275|


###### Analysis

We see that compared to `Query1Code1`, Train MRR is a bit lower (92 vs 96) but Val MRR is much lower (74 vs 85). So the ranking of precise piece of code is not as good with a single branch model which sounds logical.

Yet surprisingly the NDCG is much higher (0.17 vs 0.10) not so far from SOTA (0.19 on leaderboard). So this single encoder/tokenizer with a smaller embedding size and smaller model is much better at scoring pieces of code between each other for a query. But not as good at finding and ranking the exact right piece of code.

> What does it mean?...
>
> In the current state of my study, I can imagine that bigger BERT models have enough space to overfit the distribution and dispatch all samples (query and code) independently in the embedding space. Then using a much smaller architecture forces the model to gather/clusterize "similar" query/code samples in the same sub-space. But does it mean it unravels any semantics between query and code? It's hard to say in the current state of study and will need much deeper analysis... TO BE CONTINUED.

----

#### QueryCodeSiamese with Albert

Same configuration as previous `QueryCodeSiamese` but replacing Bert by optimized Albert model. Albert is a model described in this paper https://arxiv.org/abs/1909.11942 aimed at providing a much lighter model than Bert with the same performance. The main interesting aspect of Albert consists in introducing an intermediate internal embedding layer `E` with much lower size than hidden size `H`. This allows to reduce the number of parameters from `O(V × H)` to `O(V × E + E × H)` where `V` is the BPE Vocabulary size.

This Albert internal embedding layer seemed interesting to me in the CodeSearchNet context because it naturally introduces an embedding space compression and decompression that typically corresponds to learning atomic representation first and then extracting higher semantic representations.

##### Experiment 2020/02/18

###### Configuration

I chose an Albert model with same smaller output size 64, 256 inner embedding, 512 intermediate size and 6 layers, 8 heads.


|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query+COde Path|Albert 256/512/6/8/64|BPE 60K|

|`Training`|epochs|lr|loss|batch size|seed|epoch duration|
|---|---|---|---|---|---|---|
||10|0.0001|softmax_cross_entropy|240|0|~1h40|


###### W&B run

https://app.wandb.ai/mandubian/codenets/runs/mv433863/overview?workspace=user-mandubian

###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.8968|0.7447|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.06165**|0.03714|0.05531|0.01492|0.06287|0.07198|0.1277|


###### Analysis

MRR on Train dataset is a bit lower that with Bert and about the same on val dataset. But the NDCG is really bad in all languages.

It was quite disappointing to discover that result. I hoped it would be at least as good as classic Bert. I don't know Albert limitations enough to have clear clues on this issue. In the paper, they don't use the same tokenizer (SentencePiece) and a large batch optimizer (LAMB). I'll consider it further...

----

#### QueryCodeSiamese with LambdaLoss

The CrossEntropy used to train model by default with MRR as evaluation metric doesn't seem to be compliant with NDCG used for benchmark. NDCG takes rank and relevance into account. based on it

MRR & NDCG metrics aren't continuous so it's hard to build differentiable objective. Yet, there is a loss called `LambdaLoss` presented in this paper https://research.google/pubs/pub47258/ as a probabilistic framework for ranking metric optimization. It tries to provide a continuous, differentiable and convex rank+relevance-aware objective (in certain conditions). Converge of this loss is ensured by an EM algorithm in the paper. Mini-batch gradient descent can be seen as a kind of Expectation Maximization algorithm so we can suppose it should converge to a local minimum.

Codesearchnet dataset doesn't provide relevance of query/code samples between each other. We just know which query corresponds to which piece of code. So LambdaLoss in this context reduces to a binary ranking objective looking like MRR.

Anyway, I was curious to see if LambdaLoss was at least converging on my model in binary ranking mode.

###### Configuration

I've chosen to use smaller Vocabulary of 30K tokens and the same Bert as in the paper.

|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query+Code Path|Bert 512/6/8/128|BPE 30K|

|`Training`|epochs|lr|loss|batch size|seed|epoch duration|
|---|---|---|---|---|---|---|
||10|0.0001|lambda_loss|220|0|~1h30|

###### W&B run

https://app.wandb.ai/mandubian/codenets/runs/4nnj6vgh?workspace=user-mandubian

###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.9396|0.7992|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.1148**|0.06336|0.122|0.0684|0.09157|0.1637|0.1798|


###### Analysis

We see that the model converges but reaches lower performances than classic CrossEntropy Loss. The vocabulary and the model size weren't the same so it's hard to conclude anything robust. I can only say that used in MRR approximation (as we haven't relevance between samples), LambaLoss converges to a local optimum.

> Next step will be to compute some relevances in some way (coming soon) and retry this lambdaloss on same model compared to crossentropy.

----

#### QueryCodeSiamese with smaller embedding and Bert

So the best experiment happened with smaller embedding size.
Let's go even further with smaller embedding size of 32 and very small Bert model and larger batches.

###### Configuration

|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query+Code Path|Bert 256/2/8/32|BPE 30K|

|`Training`|epochs|lr|loss|batch size|seed|epoch duration|
|---|---|---|---|---|---|---|
||10|0.0001|cross_entropy|768|0|~30mn|

###### W&B run

https://app.wandb.ai/mandubian/codenets/runs/wz2uafe7?workspace=user-mandubian

###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.7356|0.6147|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.1721**|0.1064|0.1803|0.108|0.1508|0.2683|0.219|

###### Analysis

We see that final MRR is much lower on both train and val datasets which sounds logical. This model is not very good at finding the exact right piece of code for a query.
But the NDCG computed on benchmark is even higher than previous best bigger Bert model with bigger embedding of 64. So this model is better at ranking documents between each others.

So, the higher compression of embedding space seems to be better for NDCG and worse for MRR. I can imagine this compression forces the model to clusterize samples in the same sub-space. Something I didn't study is also the common vocabulary which also gathers all queries and language in the same tokenization space.


> I'd need to study distribution of embedding space and tokenization to find further clues.


----

#### QueryCodeSiamese with smaller embedding/model but longer token code sequences

In previous experiments, Javascript language have appeared to give the worst results in all configurations.

If we check language code distribution in [distribution notebook](https://github.com/mandubian/codenets/blob/master/codenets/codesearchnet/notebooks/codesearchnet_distrib.ipynb), we see that JS is a language that tends to be more verbose with more tokens than other languages. In previous experiments, we have trained our models with 200 max code tokens because in all languages, 200 represents the 0.9-quantile in all languages except JS. We could try to accept more code tokens and see how the model behaves (specially for JS)


###### Configuration

We take the same model with small emnedding but accept up to 400 code tokens for all languages.

|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query+Code Path|Bert 256/2/8/32|BPE 30K|

|`Training`|epochs|lr|loss|batch size|seed|epoch duration|max code tokens|
|---|---|---|---|---|---|---|---|
||10|0.0001|cross_entropy|768|0|~30mn|400|

###### W&B run

https://app.wandb.ai/mandubian/codenets/runs/e42kovab/overview?workspace=user-mandubian

###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.7356|0.6078|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.1779**|0.1065|0.1803|0.1291|0.1321|0.2485|0.2407|

###### Analysis

The result MRR is a bit lower than with less code tokens.

the mean NDCG is a bit higher than previous experiment. But if we check each language, we see that Javascript is indeed better with more code tokens (0.129 vs 0.108) but all other languages are a bit lower or the same.

So, increasing the max number of code tokens improves a bit JS results but not the others. As previous 200 max code tokens were already in the 0.9-quantile for most languages, it means adding more tokens doesn't bring much more to model training.

----

#### QueryCodeSiamese with smaller embedding/model and BPE sub-tokenization

in languages, function and symbols are often written in camelcase like `functionName` or in snakecase like `function_name`. Splitting those elements in 2 tokens `function` and `name` might help the model to extract more meaninful information from code.

So we could try to train a 30K BPE tokenizer using sub-tokenization by splitting symbol and function names.

###### Configuration

We take the same model with small emnedding but accept up to 400 code tokens for all languages.

|`Model`|Encoder|Tokenizer|
|---|---|---|
|1 Query+Code Path|Bert 256/2/8/32|BPE 30K subtokenized|

|`Training`|epochs|lr|loss|batch size|seed|epoch duration|max code tokens|
|---|---|---|---|---|---|---|---|
||10|0.0001|cross_entropy|768|0|~30mn|200|

###### W&B run

https://app.wandb.ai/mandubian/codenets/runs/5jbus5as?workspace=user-mandubian


###### Metrics

|Max MRR|Train|Val|
|---|---|---|
||0.7346|0.609|

|NDCG| Mean | Go | Java | Javascript  | Php | Python | Ruby |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|   |**0.151**|0.09706|0.2014|0.1036|0.1299|0.1894|0.1846|

###### Analysis

Results in MRR are almost the same.

NDCG is lower than both previous experiments with same model for all language except java that seems to get light advantage from this tokenizer. This is a bit surprising to me: is it due to the fact that sub-tokenization doesn't really bring anything or that smaller embedding/BERT can't take advantage from it? We need to experiment with a bigger embedding and model to check that.

----

#### Next experiments

- Use subtokenization for tokenizer
- Use Abstract Syntax Tree to represent code before code embeddings.
- Encode AST as sequences
- Encode AST graphs for embedding
- Use pretrained Bert for query embedding
- Study vocabulary and embeddings distribution

### HOCON configuration

_Incoming_

## For future, pre-Trained models ?

I'll publish my models if I can reach some good performances (this is not the case yet)
