# Methods Summary

The analysis combined NHANES 1999-2000 and 2001-2002 surplus-serum cystatin C files with demographic, laboratory, examination, and questionnaire components by participant sequence number (`SEQN`). Adults aged 20 years or older with valid creatinine and cystatin C were included; pregnant participants were excluded.

NHANES 1999-2000 creatinine was standardized as `0.147 + 1.013 × reported creatinine`. Legacy cystatin C was converted to the IFCC-traceable scale as `1.12 × (reported cystatin C - 0.12)`. The primary outcome was eGFRcys less than 70% of eGFRcr.

Continuous predictors were median-imputed and standardized; categorical predictors were most-frequent-imputed and one-hot encoded within each training pipeline. Models were evaluated using repeated stratified five-fold cross-validation and bidirectional cycle holdouts. Primary performance measures were AUROC, AUPRC, Brier score, calibration, and testing-budget yield.
