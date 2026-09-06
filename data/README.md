# External data

Raw datasets are intentionally not distributed in this repository.

## MovieLens-1M

Obtain MovieLens-1M from GroupLens and place the original files at:

```text
dataset/ml-1m/ratings.dat
dataset/ml-1m/movies.dat
dataset/ml-1m/users.dat
```

Do not substitute MovieLens Latest. Positive feedback is defined as rating >= 4. The release includes a frozen derived pilot manifest and 100-item candidate pools, but not the raw MovieLens files.

## ProRL/RecBole evaluator data

Formal-Evaluation-v1 expects the published ProRL RecBole-formatted MovieLens data under:

```text
external/ProRL/datasets/ml-1m-sas.inter
external/ProRL/datasets/ml-1m-sas.item
external/ProRL/datasets/ml-1m-sas.user
external/ProRL/datasets/ml-1m-sas.test.inter
```

Follow the provenance described in `repro/evaluators/sasrec/README.md`. These files are not redistributed.

## SASRec checkpoint

The evaluator expects:

```text
repro/results/evaluator_validation/checkpoints/SASRec-ml-1m-sas.pth
```

The checkpoint is intentionally excluded. The loader records the pinned ProRL source URL and verifies downloaded bytes against local metadata when prepared in the original environment.

