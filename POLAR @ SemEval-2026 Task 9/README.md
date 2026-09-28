# POLAR @ SemEval-2026 Task 9

This folder contains my experiments for **POLAR @ SemEval-2026 Task 9**, a shared task on multilingual online polarization.

The work covers **English and Hausa** across the three subtasks, with most of the later experimentation focused on **Subtask 3**, which also led to an ACL/SemEval 2026 system paper.

## Paper

**DeepSemantics at SemEval-2026 Task 9: Label-Wise Optimization with Adaptive Focal Loss for Polarization Manifestation Identification**

ACL Anthology:  
https://aclanthology.org/2026.semeval-1.210/

The paper focuses on **Subtask 3** and explores label-wise optimization under severe class imbalance using:

- RoBERTa-base for English
- Afro-XLM-R-small for Hausa
- One-vs-Rest classification
- controlled oversampling
- Adaptive Focal Loss
- label-specific decision thresholds

## Task overview

### Subtask 1 — Polarization Detection
Binary classification:

- polarized
- not polarized

Notebooks:
- [`subtask_1_english.ipynb`](subtask_1_english.ipynb)
- [`subtask_1_haussa.ipynb`](subtask_1_haussa.ipynb)

### Subtask 2 — Polarization Type Classification
Multi-label classification of the polarization target:

- Political
- Racial/Ethnic
- Religious
- Gender/Sexual
- Other

Notebooks:
- [`subtask_2_english.ipynb`](subtask_2_english.ipynb)
- [`subtask_2_haussa.ipynb`](subtask_2_haussa.ipynb)

### Subtask 3 — Polarization Manifestation Identification
Multi-label classification of how polarization is expressed:

- Stereotype
- Vilification
- Dehumanization
- Extreme Language
- Lack of Empathy
- Invalidation

Notebooks:
- [`subtask_3_english.ipynb`](subtask_3_english.ipynb)
- [`subtask_3_hausa.ipynb`](subtask_3_hausa.ipynb)

## Notes

The notebooks contain both baseline experiments and later imbalance-aware approaches such as class weighting, focal loss, oversampling, and threshold optimization.

The Subtask 3 paper represents the final system; the notebooks keep part of the experimentation path that led to it.
