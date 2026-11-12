# Computational Psychiatry of Anxiety: Behavioral & Cognitive Modeling Pipeline

This directory contains the behavioral preprocessing, natural language processing (NLP) on memory recall, and hierarchical Bayesian cognitive modeling pipeline to investigate how generalized anxiety symptoms (measured via GAD-7) relate to learning and decision-making.

---

## 1. Sequential Notebook Flow

The notebooks are organized in a sequential, step-by-step pipeline from raw data preprocessing to advanced model comparison:

1. **[01_process_behavioral_data.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/01_process_behavioral_data.ipynb)**
   - **Purpose:** Preprocesses raw behavioral data and survey data. Drops missing choices, maps stimulus IDs to integer codes, scales trial rewards, and compiles the trial-by-trial combined dataset (`allsubs_combined.csv`).
   *needs manual verification.

2. **[02_free_recall_sentiment_analysis.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/02_free_recall_sentiment_analysis.ipynb)**
   - **Purpose:** Runs VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis on free-recall textual data written by participants during the memory retrieval task.

3. **[02_free_recall_extremes.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/02_free_recall_extremes.ipynb)**
   - **Purpose:** Sentiment analysis and GAD-7 correlation specifically filtering for and keeping only the extreme bottom 6 and top 6 subjects based on GAD-7 anxiety scores.

4. **[03_bayesian_nlp_memory_recall.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/03_bayesian_nlp_memory_recall.ipynb)**
   - **Purpose:** Fits a Bayesian linear regression model to examine if free-recall memory properties (specifically, the sentiment score of recalled items) interact with subject-level GAD-7 anxiety symptoms.

5. **[04_bayesian_robust_regression.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/04_bayesian_robust_regression.ipynb)**
   - **Purpose:** Employs robust linear regression (using Student-T likelihood to accommodate outliers) to evaluate whether Day 1 behavioral parameters credibly predict Day 2 learning performance.

6. **[05_bayesian_rl_hierarchical_comparison.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/05_bayesian_rl_hierarchical_comparison.ipynb)**
   - **Purpose:** Formulates and compares three hierarchical reinforcement learning (RL) models (Single Alpha, Valence, and Task-domain models) using ArviZ's LOO cross-validation.

7. **[06_bayesian_rl_learning_rates_vs_gad7.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/06_bayesian_rl_learning_rates_vs_gad7.ipynb)**
   - **Purpose:** Restricts the winning RL model (Valence Model) by parameterizing positive and negative learning rates directly as log-linear functions of GAD-7 scores, isolating how anxiety symptoms influence reward vs. loss learning rates.

8. **[07_bayesian_kalman_filter_vs_gad7.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/07_bayesian_kalman_filter_vs_gad7.ipynb)**
   - **Purpose:** Implements a Hierarchical Bayesian Kalman Filter on the bandit task, testing whether GAD-7 anxiety scores predict subject-level belief drift (Process Noise $Q$) and observation uncertainty (Observation Noise $R$).

9. **[08_model_comparison_rl_vs_kalman.ipynb](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/08_model_comparison_rl_vs_kalman.ipynb)**
   - **Purpose:** Performs a formal model comparison using LOO cross-validation between the Rescorla-Wagner (Valence RL) model and the standard Kalman Filter model (non-GAD-parameterized) to determine which mathematical framework best explains participant choices.

---

## 2. Supporting Python Scripts

### Modeling & Utility Helper

- **[learning_functions.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/learning_functions.py):** Contains the PyTensor scan step functions for reinforcement learning models (`update_q_hierarchical`, `update_m2_hierarchical`, and `update_m3_hierarchical`).

### Data Pipeline Scripts

- **[process_data.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/process_data.py):** Processes initial raw behavioral logs.
- **[merge_data.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/merge_data.py):** Merges Day 1 behavior, Day 2 behavior, and survey scores.
- **[process_survey.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/process_survey.py):** Cleans GAD-7 and other survey questions.
- **[process_day2.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/process_day2.py) & [process_day2_correct.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/process_day2_correct.py):** Preprocesses Day 2 outcome variables.
- **[extract_free_recall.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/extract_free_recall.py) & [run_vader_nlp.py](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/run_vader_nlp.py):** Sentiment and NLP feature extraction utilities.

---

## 3. Rebuilding the Notebooks

The python scripts used to programmatically generate and update these Jupyter notebooks are housed in the **[notebook_builders/](file:///Users/nachshonkorem/Documents/AntiGrav/Anxiety/code/notebook_builders/)** folder.

To regenerate any notebook, run its respective builder script from this root directory:

```bash
python notebook_builders/build_kalman_notebook.py
python notebook_builders/build_compare_notebook.py
python notebook_builders/build_sentiment_analysis_extremes_notebook.py
```

*(All builder scripts have been updated to write directly to the new sequential notebook filenames in this directory.)*
