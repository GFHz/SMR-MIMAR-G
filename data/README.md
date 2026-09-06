# Data

This repository does not redistribute MovieLens-1M. Download MovieLens-1M from the official GroupLens source and place the original files at:

```text
dataset/ml-1m/ratings.dat
dataset/ml-1m/movies.dat
dataset/ml-1m/users.dat
```

Do not substitute MovieLens Latest. A positive interaction is a rating `>= 4`. The final controlled comparison uses each selected user's complete chronological positive history.

Formal-Evaluation-v1 additionally expects external ProRL/RecBole MovieLens files under `external/ProRL/datasets/` and the public SASRec checkpoint at `repro/results/evaluator_validation/checkpoints/SASRec-ml-1m-sas.pth`. See `repro/evaluators/sasrec/README.md`. These assets are not redistributed.
