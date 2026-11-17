# Computational Psychiatry of Anxiety: Behavioral, NLP & Cognitive Modeling Pipeline

This repository contains the behavioral preprocessing, natural language processing (NLP) free-recall sentiment analysis, and hierarchical Bayesian cognitive modeling pipeline used to investigate how generalized anxiety symptoms (measured via GAD-7) relate to reinforcement learning, belief updating, and memory retrieval.

All Bayesian cognitive models are optimized for **PyMC v6 (`pymc 6.0+`, `pytensor 3.0+`, `arviz 1.1+`)** using the Rust-based **`nutpie`** sampler (`nuts_sampler="nutpie"`) and subject-vectorized 2D `pytensor.scan` loops (`100` trial steps across all `67` participants in parallel).

---

## 1. Sequential Notebook Flow

The notebooks are organized in a sequential pipeline from raw data preprocessing to hierarchical Bayesian model comparison:

1. **`01_process_behavioral_data.ipynb`**
   - **Purpose:** Preprocesses raw Day 1 four-armed bandit behavioral logs, applies quality-control exclusions (removing reversal and incomplete subjects), and compiles `../data/allsubs_day1.csv`.

2. **`02_free_recall_sentiment_analysis.ipynb`**
   - **Purpose:** Runs VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis on free-recall narratives collected before and after the task on Day 1 and Day 2 (`../data/free_recall_nlp.csv`), testing within-subject valence/time shifts and continuous correlations with GAD-7 across the full sample.

3. **`02_free_recall_extremes.ipynb` & `anxiety_comparison.ipynb`**
   - **Purpose:** Mixed-ANOVA and group comparisons of free-recall sentiment shifts comparing low-anxiety (`GAD-7 < 4`) vs. high-anxiety (`GAD-7 > 10`) participants across timepoints and valences.

4. **`02_free_recall_day2pre_vs_day1post.ipynb`**
   - **Purpose:** Analyzes the overnight consolidation shift in free-recall sentiment from the end of Day 1 (**Day 1 Post**) to the start of Day 2 (**Day 2 Pre**), including full-sample paired $t$-tests, GAD-7 correlations, extreme-group Mixed ANOVAs, and clinical-severity power calculations.

5. **`03_bayesian_nlp_memory_recall.ipynb`**
   - **Purpose:** Fits a Bayesian robust linear regression model (Student-T likelihood via `nutpie`) to test whether Day 1 free-recall sentiment shifts predict Day 2 Win and Lose memory accuracy.

6. **`04_bayesian_robust_regression.ipynb`**
   - **Purpose:** Employs Bayesian robust linear regression (Student-T likelihood via `nutpie`) to evaluate whether subject-level GAD-7 scores predict Day 2 Win and Lose memory performance.

7. **`05_bayesian_rl_hierarchical_comparison.ipynb`**
   - **Purpose:** Fits and compares four hierarchical Rescorla-Wagner reinforcement learning (RL) models using PyMC v6 (`nutpie` NUTS sampler, vectorized across participants) and ArviZ Leave-One-Out (LOO) cross-validation:
     - **Model 1 (`Single Alpha`):** Single learning rate ($\alpha$) and inverse temperature ($\beta$).
     - **Model 2 (`Valence`):** Separate learning rates for positive ($\alpha_{\text{pos}}$) and negative ($\alpha_{\text{neg}}$) prediction errors, plus $\beta$.
     - **Model 3 (`Task`):** Separate learning rates ($\alpha_{\text{win}}, \alpha_{\text{lose}}$) and inverse temperatures ($\beta_{\text{win}}, \beta_{\text{lose}}$) by task domain.
     - **Model 4 (`Task x Valence`):** Four learning rates split by both task domain and prediction error valence ($\alpha_{\text{win,pos}}, \alpha_{\text{win,neg}}, \alpha_{\text{lose,pos}}, \alpha_{\text{lose,neg}}$), plus $\beta$.

8. **`06_bayesian_rl_learning_rates_vs_gad7.ipynb`**
   - **Purpose (Constrained Direct-Effect GAD-7 2-Alpha Valence Model):** Restricts between-subject variation in the 2-alpha Valence RL model so that **only** a participant's GAD-7 score can change $\alpha_{\text{pos}}$ and $\alpha_{\text{neg}}$ across participants (`alpha_i = invlogit(intercept + slope * GAD7_norm_i)`).

9. **`06b_subject_specific_alphas_gad7_variance.ipynb`**
   - **Purpose (Subject-Specific 2-Alpha Valence & Variance Explained by GAD-7):** Estimates **subject-specific** $\alpha_{\text{pos}}$ and $\alpha_{\text{neg}}$ by including hierarchical individual random effects alongside GAD-7 (`logit(alpha_i) = intercept + slope * GAD7_norm_i + z_i * sigma`), quantifying the variance ($R^2$) explained by GAD-7 and the correlation between $\alpha_{\text{pos}}$ and $\alpha_{\text{neg}}$.

10. **`16_bayesian_rl_4alphas_vs_gad7.ipynb`**
    - **Purpose (Constrained Direct-Effect GAD-7 4-Alpha Task $\times$ Valence Model):** Restricts between-subject variation in the winning 4-learning-rate model (`Task x Valence`) so that **only** a participant's GAD-7 score can change $\alpha_{\text{win,pos}}, \alpha_{\text{win,neg}}, \alpha_{\text{lose,pos}}, \alpha_{\text{lose,neg}}$ across participants.

11. **`16b_subject_specific_4alphas_gad7_variance.ipynb`**
    - **Purpose (Subject-Specific 4-Alpha Task $\times$ Valence & Variance Explained by GAD-7):** Estimates **subject-specific** $\alpha_{\text{win,pos}}, \alpha_{\text{win,neg}}, \alpha_{\text{lose,pos}}, \alpha_{\text{lose,neg}}$ with hierarchical random effects alongside GAD-7, quantifying the variance ($R^2$) explained by GAD-7 for each of the four learning rates and plotting the $4 \times 4$ subject-level correlation matrix.

12. **`07_bayesian_kalman_filter_vs_gad7.ipynb`**
    - **Purpose:** Implements a vectorized Hierarchical Bayesian Kalman Filter on the bandit task (`nutpie`), testing whether GAD-7 anxiety scores predict subject-level belief drift (Process Noise $Q$) and observation uncertainty (Observation Noise $R$).

13. **`08_model_comparison_rl_vs_kalman.ipynb`**
    - **Purpose:** Performs a formal LOO cross-validation model comparison between the hierarchical Rescorla-Wagner (Valence) model and the hierarchical Kalman Filter model to determine which computational framework better accounts for participant choices.

---

## 2. Supporting Python Scripts

### Modeling Utilities
- **`learning_functions.py`:** Vectorized PyTensor `scan` step functions (`prepare_vectorized_bandit_data`, `step_m1_vec`, `step_m2_vec`, `step_m3_vec`, `step_m4_vec`, `step_kalman_vec`) and PyMC v6 / ArviZ 1.1+ `DataTree` helpers (`add_bernoulli_ll`, `get_hdi_bounds`, `plot_posterior_hdi`).

### Data Preprocessing Scripts
- **`process_data.py`:** Processes raw Day 1 behavioral trial logs into `../data/allsubs_day1.csv`.
- **`process_day2.py` & `process_day2_correct.py`:** Processes Day 2 memory test logs and computes per-subject Win/Lose recall accuracy (`../data/allsubs_day2.csv`).
- **`process_survey.py`:** Cleans Qualtrics assessment logs and computes total GAD-7 scores (`../data/survey.csv`).
- **`merge_data.py`:** Merges Day 1 trial-by-trial behavior, Day 2 memory accuracy, and GAD-7 survey scores into `../data/allsubs_combined.csv`.
- **`extract_free_recall.py` & `run_vader_nlp.py`:** Extracts Day 1 and Day 2 pre/post free-recall text responses (`../data/free_recall.csv`) and computes VADER sentiment scores (`../data/free_recall_nlp.csv`).
