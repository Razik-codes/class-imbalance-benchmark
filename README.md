# Studying Class Imbalance in Image and Tabular Classification

This repository documents an individual class project in which I explored a practical question that comes up often in applied machine learning: **how do simple, commonly used classifiers behave when the training data become progressively more imbalanced?**

I built the experiments to compare straightforward mitigation strategies across small tabular data (Iris) and image-based benchmarks (MNIST and CIFAR-10). My goal was not to present a new method, but to develop a careful, reproducible baseline and to examine the trade-offs behind aggregate performance numbers.

## Project scope

For each dataset, I create controlled class distributions in the training split while keeping a held-out test split fixed. I evaluate:

- Logistic regression trained on the imbalanced data
- Logistic regression after random oversampling
- Logistic regression with `class_weight="balanced"`
- A linear-booster XGBoost baseline

The notebooks report accuracy, precision, recall, and F1 for binary tasks, and accuracy plus weighted F1 for multiclass tasks. Binary label construction is explicit in the relevant notebooks: Iris groups classes 0–1 versus class 2, while MNIST and CIFAR-10 group labels 0–4 versus 5–9.

## Notebooks

| Dataset | Binary experiment | Multiclass experiment |
| --- | --- | --- |
| Iris | `iris_binary_14004110.ipynb` | `iris_multiclass_14004110.ipynb` |
| MNIST | `mnist_binary_14004110.ipynb` | `mnist_multiclass_14004110.ipynb` |
| CIFAR-10 | `cifar10_binary_14004110.ipynb` | `cifar10_multiclass_14004110.ipynb` |

The MNIST notebooks retrieve data through OpenML; the CIFAR-10 notebooks use TensorFlow/Keras' dataset loader. Iris is bundled with scikit-learn, so no separate download is needed for that experiment.

## What I learned

In the binary Iris and MNIST experiments, the most severe imbalance settings make the difference between a reasonable-looking accuracy number and a useful classifier especially clear. Random oversampling and class weighting often recovered more balanced performance than training directly on the skewed data. At more moderate imbalance levels, the choice of strategy involved real precision/recall and F1 trade-offs rather than a universally best option.

I treat these results as exploratory coursework, not as a statistically conclusive benchmark: each setting uses one train/test split and the image models are deliberately simple linear baselines. The saved notebook outputs also retain convergence warnings from the original runs. These limitations are useful reminders for future work: use stratified repeated splits, report class-wise or macro metrics, tune the models consistently, and evaluate stronger image representations.

## Reproducibility

The original notebooks were developed with Python 3.9. To set up a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks from top to bottom. Dataset downloads are requested on first use and are not stored in this repository. Results can vary slightly across library versions and data-source revisions.

## Tools

Python, NumPy, scikit-learn, imbalanced-learn, XGBoost, TensorFlow/Keras, and Jupyter.

## Authorship

I completed this repository as an individual class project. It is included in my portfolio to show my early hands-on work with experimental design, class imbalance, model evaluation, and reproducible computational notebooks.

## License

This project is available under the [MIT License](LICENSE).
