# Computational Psychiatry of Anxiety: Behavioral, NLP & Cognitive Modeling Pipeline

This repository contains the behavioral preprocessing, natural language processing (NLP) free-recall sentiment analysis, and hierarchical Bayesian cognitive modeling pipeline used to investigate how generalized anxiety symptoms (measured via GAD-7) relate to reinforcement learning, belief updating, and memory retrieval.

---

## 1. Sequential Notebook Flow

The notebooks are organized in a sequential pipeline from raw data preprocessing to hierarchical Bayesian model comparison:

1. **`01_process_behavioral_data.ipynb`**
   - **Purpose:** Preprocesses raw Day 1 four-armed bandit behavioral logs, applies quality-control exclusions (removing reversal and incomplete subjects), and compiles `../data/allsubs_day1.csv`.

2. **`02_free_recall_sentiment_analysis.ipynb`**
   - **Purpose:** Runs VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis on free-recall narratives collected before and after the task on Day 1 and Day 2 (`../data/free_recall_nlp.csv`), testing within-subject valence/time shifts and continuous correlations with GAD-7 across the full sample.

3. **`02_free_recall_extremes.ipynb` & `anxiety_comparison.ipynb`**
   - **Purpose:** Mixed-ANOVA and group comparisons of free-recall sentiment shifts comparing low-anxiety (`GAD-7 < 4`) vs. high-anxiety (`GAD-7 > 10`) participants across timepoints and valences.

4. **`03_bayesian_nlp_memory_recall.ipynb`**
   - **Purpose:** Fits a Bayesian robust linear regression model (Student-T likelihood) to test whether Day 1 free-recall sentiment shifts predict Day 2 Win and Lose memory accuracy.

5. **`04_bayesian_robust_regression.ipynb`**
   - **Purpose:** Employs Bayesian robust linear regression (Student-T likelihood) to evaluate whether subject-level GAD-7 scores predict Day 2 Win and Lose memory performance.

6. **`05_bayesian_rl_hierarchical_comparison.ipynb`**
   - **Purpose:** Fits and compares three hierarchical Rescorla-Wagner reinforcement learning (RL) models using PyMC (`numpyro` NUTS sampler) and ArviZ Leave-One-Out (LOO) cross-validation:
     - **Model 1 (`Single Alpha`):** Single learning rate ($\alpha$) and inverse temperature ($\beta$).
     - **Model 2 (`Valence`):** Separate learning rates for positive ($\alpha_{\text{pos}}$) and negative ($\alpha_{\text{neg}}$) prediction errors, plus $\beta$.
     - **Model 3 (`Task`):** Separate learning rates ($\alpha_{\text{win}}, \alpha_{\text{lose}}$) and inverse temperatures ($\beta_{\text{win}}, \beta_{\text{lose}}$) by task domain.

7. **`06_bayesian_rl_learning_rates_vs_gad7.ipynb`**
   - **Purpose:** Extends the winning Valence RL model by parameterizing $\alpha_{\text{pos}}$ and $\alpha_{\text{neg}}$ on the logit scale as linear functions of normalized GAD-7 scores, directly testing how anxiety modulates learning from positive vs. negative prediction errors.

8. **`07_bayesian_kalman_filter_vs_gad7.ipynb`**
   - **Purpose:** Implements a Hierarchical Bayesian Kalman Filter on the bandit task, testing whether GAD-7 anxiety scores predict subject-level belief drift (Process Noise $Q$) and observation uncertainty (Observation Noise $R$).

9. **`08_model_comparison_rl_vs_kalman.ipynb`**
   - **Purpose:** Performs a formal LOO cross-validation model comparison between the hierarchical Rescorla-Wagner (Valence) model and the hierarchical Kalman Filter model to determine which computational framework better accounts for participant choices.

---

## 2. Supporting Python Scripts

### Modeling Utilities
- **`learning_functions.py`:** PyTensor `scan` step functions for the hierarchical reinforcement learning models (`update_q_hierarchical`, `update_m2_hierarchical`, and `update_m3_hierarchical`).

### Data Preprocessing Scripts
- **`process_data.py`:** Processes raw Day 1 behavioral trial logs into `../data/allsubs_day1.csv`.
- **`process_day2.py` & `process_day2_correct.py`:** Processes Day 2 memory test logs and computes per-subject Win/Lose recall accuracy (`../data/allsubs_day2.csv`).
- **`process_survey.py`:** Cleans Qualtrics assessment logs and computes total GAD-7 scores (`../data/survey.csv`).
- **`merge_data.py`:** Merges Day 1 trial-by-trial behavior, Day 2 memory accuracy, and GAD-7 survey scores into `../data/allsubs_combined.csv`.
- **`extract_free_recall.py` & `run_vader_nlp.py`:** Extracts Day 1 and Day 2 pre/post free-recall text responses (`../data/free_recall.csv`) and computes VADER sentiment scores (`../data/free_recall_nlp.csv`).
