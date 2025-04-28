# String Matching Algorithms: Horspool vs Boyer-Moore

This project compares the performance of **Horspool's** and **Boyer-Moore's** string matching algorithms.

We test:
- Real-world data using the **IMDB Movie Review** dataset
- Synthetic data with varying levels of **noise**

Both **accuracy** and **efficiency** are measured.

---

## Files

- `main.py` — Tests on IMDB data
- `main_synthetic.py` — Tests on synthetic data
- `synthetic_noise_experiment.py` — Tests noise on synthetic data
- `horspool.py` — Horspool algorithm
- `boyer_moore.py` — Boyer-Moore algorithm
- `download_IMDB.py` — IMDB dataset downloader
- `datasets/` — Experiment results and graphs

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## How to Run
First run ```python3 main.py```. This will download the imdb dataset and extract it for you. 
This will also run the analysis for the imdb dataset.
```bash
# Run IMDB experiments
python main.py
```
To run a prelimenary test on synthetic data: run ```python3 main_synthetic.py```.
```bash
# Run synthetic experiments
python main_synthetic.py
```
To run an analysis for noise on syntheticly generated data: run ```python3 synthetic_noise_experiment.py```.
```bash
# Run synthetic noise experiment
python synthetic_noise_experiment.py
```


