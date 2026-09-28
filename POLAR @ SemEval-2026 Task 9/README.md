# Hate Speech Detection – Subtasks (English & Hausa)

This repository contains the code and experiments for hate speech detection on **English and Hausa** data, focusing on the different subtasks defined in the shared task.

The project is implemented using Transformer-based models and Jupyter notebooks for training, evaluation, and analysis.

---

## Subtasks Description

- **Subtask 1**: Binary classification (hate vs. non-hate).
- **Subtask 2**: Multi-label classification for hate categories.
- **Subtask 3**: Fine-grained classification of hate speech targets.

Each subtask is handled separately to better account for label distribution and task-specific characteristics.

---

## Languages

- **English**
- **Hausa**

The Hausa setting is particularly challenging due to limited annotated resources and linguistic variability.

---

## Repository Structure


---

## Running the Notebooks

The notebooks can be opened and executed using **Google Colab**:

- Subtask 1 (English): [Open in Colab](https://drive.google.com/file/d/1fcYtkazva0OFRz2WE2zPctdsLozOykgg/view?usp=drive_link)
- Subtask 1 (Hausa): [Open in Colab](https://drive.google.com/file/d/1lTZyzt22dMU41fpl6WLbxuFIyujDoTYK/view?usp=drive_link)
- Subtask 2: [Open in Colab](https://drive.google.com/file/d/1OTXYCVWszMLjEHz25CKYkTvlLcAvbTWv/view?usp=drive_link)
- Subtask 3: [Open in Colab](https://colab.research.google.com/drive/10AqrS0zmqApG7dnHbnjEkSUK8BumZrPA?usp=drive_link)

---

## Notes

- Outputs have been cleared before uploading to keep the repository lightweight.
- Class imbalance is handled during training using weighted loss functions.
- Evaluation is reported using standard metrics (Precision, Recall, F1-score).

---


---

## Acknowledgements

This work is part of a hate speech detection study involving low-resource languages, with a particular focus on Hausa.
