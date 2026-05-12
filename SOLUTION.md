# SMILES-2026 Hallucination Detection Solution

## Reproducibility

The repository is self-contained and uses the provided `solution.py` entry point.
I did not change the fixed infrastructure files (`solution.py`, `evaluate.py`,
or `model.py`). The implemented parts are:

- `aggregation.py`
- `probe.py`
- `splitting.py`

To reproduce the run from a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python solution.py
```

Running `python solution.py` produces:

- `results.json`
- `predictions.csv`

The submitted `predictions.csv` has the required format:

```csv
id,label
0,1
1,1
...
```

## Final Approach

The final solution is a lightweight linear probe on top of hidden-state features
from Qwen2.5-0.5B. The main goal was to keep the method simple and reproducible,
while still using more information than only the final token from the final
layer.

### Feature aggregation

In `aggregation.py`, I use four late transformer layers:

- final layer: `-1`
- previous layer: `-2`
- layer `-4`
- layer `-8`

For each selected layer, I extract three views of the sequence:

- the last real token representation
- the mean representation over the full non-padding sequence
- the mean representation over the last 32 real tokens

These vectors are concatenated into one feature vector. Since Qwen2.5-0.5B has a
hidden size of 896, the final feature size is:

```text
4 layers * 3 pooling views * 896 hidden dimensions = 10752 features
```

This worked better than relying only on the final layer or only on the last
token. The intuition is that hallucination signals can appear across nearby
late layers and across the generated response context, not necessarily in one
single representation.

I kept `USE_GEOMETRIC = False`, so no additional geometric features are appended
in the final run.

### Classifier

In `probe.py`, I use:

- `StandardScaler` for feature normalization
- `LogisticRegression`
- `C=0.1`
- `class_weight="balanced"`
- `solver="liblinear"`
- `random_state=42`

I originally kept the skeleton MLP structure available in the class, but the
final fitted model is logistic regression. With only 689 labelled samples and a
high-dimensional feature vector, the linear model was more stable and easier to
regularize than a neural classifier.

The decision threshold is tuned on the validation split. I evaluate candidate
thresholds from the predicted probabilities plus a coarse grid from 0.0 to 1.0.
The selected threshold maximizes validation accuracy, with F1 used as a
tie-breaker.

### Splitting

In `splitting.py`, I use 5-fold `StratifiedKFold` with `random_state=42`. Inside
each fold, I split the train/validation part again to create a validation split
for threshold tuning. Stratification preserves the label balance across train,
validation, and internal test subsets.

## Results

The current `results.json` was produced by the `solution.py` script.

Summary:

| Metric | Value |
| --- | ---: |
| Number of labelled samples | 689 |
| Number of folds | 5 |
| Feature dimension | 10752 |
| Average baseline accuracy | 0.7010 |
| Average probe train accuracy | 0.8729 |
| Average probe validation accuracy | 0.7408 |
| Average probe test accuracy | 0.7097 |
| Average probe test F1 | 0.8204 |
| Average probe test AUROC | 0.6836 |

The internal test accuracy is slightly above the majority-class baseline. The
validation accuracy is higher than the internal test accuracy, so I treat the
final score as useful but not overclaiming. The train AUCROC is 1.0, which is a
sign that the high-dimensional features can overfit the small labelled dataset;
this is why I used a regularized linear model and validation threshold tuning
instead of a larger neural classifier.

## What Helped Most

The biggest improvement came from using several late layers and several pooling
views at once. The single final-token representation was too narrow, while
combining last-token, full-sequence mean, and tail mean gave the probe a better
summary of both the answer endpoint and the broader generated context.

The second important choice was using a regularized logistic regression model
instead of a larger MLP. The dataset is small, and the feature vector is already
rich, so a simple classifier was a better fit.

## Experiments and Discarded Ideas

I tried or considered the following alternatives:

- Final-layer-only features. This was simpler but less informative than the
  multi-layer representation.
- Last-token-only pooling. This lost information from the response body and was
  less stable than combining multiple pooling views.
- Neural MLP probe. It was more flexible, but with the available dataset size it
  was easier to overfit and harder to make stable across splits.
- Additional hand-crafted geometric features. I kept them out of the final
  solution because the multi-layer pooled hidden states already produced a large
  feature vector, and adding more features did not seem worth the extra
  complexity for this run.
- Single train/validation/test split. I switched to 5-fold stratified evaluation
  to get a more reliable estimate on such a small dataset.

## Final Notes

The final submission files are:

- `results.json`, generated by `python solution.py`
- `predictions.csv`, generated by `python solution.py`
- `SOLUTION.md`, this report

The repository can be evaluated by running the original `solution.py` without
manual changes to the fixed infrastructure.
