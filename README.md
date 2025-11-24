# 🧠 xTurnovers_Models: Reproducible Framework for Expected Pass Turnovers (xPT)

**Author:** Andrew Peters (Middlesex University & Leicester City Football Club)
**Contact:** [andrewpeters1994@gmail.com](mailto:andrewpeters1994@gmail.com)
**Associated paper:** *Data Leakage and Predictive Validity in Machine Learning Models of Pass Turnovers* (Peters et al., 2025)

---

## 📘 Overview

This repository provides the **fully reproducible analytical pipeline** used in the study:

> **“Data Leakage and Predictive Validity in Machine Learning Models of Pass Turnovers” (Peters et al., 2025)**

The project extends the *Expected Pass Turnovers (xPT)* framework to evaluate the impact of **data leakage** and **temporal validity** in football turnover prediction models.

Sample dataset has been randomly generated and anonymised: [sample_data.csv](sample_data/sample_data.csv)

---

## ⚽ Research Summary

The **xPT** framework models the likelihood of a pass leading to a turnover, comparing **default (leakage-inclusive)** and **alternative (leakage-corrected)** feature sets across four algorithms:

1. Mixed-effects logistic regression
2. Penalised logistic regression
3. Random forest
4. Gradient boosting (XGBoost)

Findings show that excluding post-execution features reduces AUC by ~0.13 on average, but ensures temporal validity for real-time tactical applications.

---

## 🧩 Repository Structure

```
xTurnovers_Models/
│
├── sample_data/
│   └── initial_data.subset.all.csv
│
├── paper_outputs/
│   ├── output_default/
│   ├── output_alt/
│
├── scripts/
│   ├── turnover_pipeline_run.R
│   ├── turnover_pipeline.R
│   ├── turnover_evaluation_suite.R
│   └── run_machine_learning_pipeline.sh
│
├── figures/
│   ├── combined_calibration_plot.png
│   ├── combined_shap.png
│   ├── combined_confusion_matrix.png
│   ├── combined_pdp.png
│   ├── combined_auc_plot.png
│
├── README.md
└── requirements.txt
```

---

## 🧠 Reproducibility

To reproduce analysis on the provided dataset:

```bash
# Default (leakage-inclusive)
./run_machine_learning_pipeline.sh sample_data/sample_data.csv all_output default

# Alternative (leakage-corrected)
./run_machine_learning_pipeline.sh sample_data/sample_data.csv all_output alt
```

---

## 📊 Model Performance Summary

| Algorithm              | Default AUC | Leakage-Corrected AUC | ΔAUC   |
| ---------------------- | ----------- | --------------------- | ------ |
| Mixed-effects logistic | 0.789       | 0.707                 | -0.082 |
| Penalised logistic     | 0.786       | 0.690                 | -0.096 |
| Random forest          | 0.920       | 0.737                 | -0.183 |
| Gradient boosting      | 0.924       | 0.742                 | -0.182 |

---

## 📂 Core Figures

![AUC Plots](figures/combined_auc_plot.png)

![Calibration Plots](figures/combined_calibration_plot.png)

![SHAP Plots](figures/combined_shap.png)

![Confustion Matrix](figures/combined_confusion_matrix.png)

![PDPPlots](figures/combined_pdp.png)

---

## 📘 Citation

If you use this repository, please cite:

**Peters, A., Parmar, N., Davies, M., & James, N. (Pending Publication).**
*Data Leakage and Predictive Validity in Machine Learning Models of Pass Turnovers.*
*Journal of Sports Sciences.*

and

**Peters, A., Parmar, N., Davies, M., & James, N. (2024).**
*Expected Pass Turnovers (xPT): A model to analyse turnovers from passing events in football.*
*Journal of Sports Sciences, 42(10), 1234–1245.*

---

## ⚖️ License

Released under the **MIT License**.
StatsBomb data are proprietary and excluded from redistribution.

---

## 💬 Contact

For questions or collaborations, please contact:
**[andrewpeters1994@gmail.com](mailto:andrewpeters1994@gmail.com)**
or open an issue in this repository.
