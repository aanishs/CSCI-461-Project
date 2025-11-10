# References

**Project:** Tic Episode Prediction using Machine Learning
**Course:** CSCI-461 Machine Learning
**Date:** November 2025

---

## Software Libraries and Frameworks

### Machine Learning

1. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12(Oct), 2825-2830.
   - **Used for:** Random Forest implementation, model evaluation, cross-validation (RandomizedSearchCV)
   - **Version:** scikit-learn 1.3+
   - **URL:** https://scikit-learn.org/

2. **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).
   - **Used for:** Gradient boosting classifier and regressor
   - **Version:** xgboost 2.0+
   - **URL:** https://xgboost.readthedocs.io/
   - **DOI:** 10.1145/2939672.2939785

3. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).** LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.
   - **Used for:** Alternative gradient boosting implementation (planned for future work)
   - **Version:** lightgbm 4.0+
   - **URL:** https://lightgbm.readthedocs.io/

### Data Processing

4. **McKinney, W. (2010).** Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51-56).
   - **Used for:** Data manipulation, feature engineering, time-series operations
   - **Library:** pandas
   - **Version:** pandas 2.0+
   - **URL:** https://pandas.pydata.org/

5. **Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020).** Array programming with NumPy. *Nature*, 585(7825), 357-362.
   - **Used for:** Numerical computing, array operations, statistical functions
   - **Library:** numpy
   - **Version:** numpy 1.24+
   - **URL:** https://numpy.org/
   - **DOI:** 10.1038/s41586-020-2649-2

### Visualization

6. **Hunter, J. D. (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.
   - **Used for:** Figure generation, model architecture diagrams, performance plots
   - **Library:** matplotlib
   - **Version:** matplotlib 3.7+
   - **URL:** https://matplotlib.org/
   - **DOI:** 10.1109/MCSE.2007.55

7. **Waskom, M. L. (2021).** seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
   - **Used for:** Statistical visualizations, heatmaps, distribution plots
   - **Library:** seaborn
   - **Version:** seaborn 0.12+
   - **URL:** https://seaborn.pydata.org/
   - **DOI:** 10.21105/joss.03021

---

## Machine Learning Algorithms and Methods

### Ensemble Methods

8. **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5-32.
   - **Used for:** Regression task (next tic intensity prediction)
   - **Algorithm:** Random Forest with bootstrap aggregating (bagging)
   - **Implementation:** scikit-learn RandomForestRegressor and RandomForestClassifier
   - **DOI:** 10.1023/A:1010933404324

9. **Friedman, J. H. (2001).** Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
   - **Used for:** Theoretical foundation for gradient boosting
   - **Algorithm:** Gradient Boosting Decision Trees (GBDT)
   - **DOI:** 10.1214/aos/1013203451

### Hyperparameter Optimization

10. **Bergstra, J., & Bengio, Y. (2012).** Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(1), 281-305.
   - **Used for:** RandomizedSearchCV for hyperparameter tuning
   - **Method:** Random sampling from hyperparameter distributions
   - **URL:** https://www.jmlr.org/papers/v13/bergstra12a.html

### Cross-Validation

11. **Kohavi, R. (1995).** A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence* (Vol. 14, No. 2, pp. 1137-1145).
   - **Used for:** K-fold cross-validation (k=3) with user-grouped stratification
   - **Method:** Group-based cross-validation to prevent data leakage

---

## Evaluation Metrics

### Regression Metrics

12. **Willmott, C. J., & Matsuura, K. (2005).** Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. *Climate Research*, 30(1), 79-82.
   - **Metric:** Mean Absolute Error (MAE)
   - **Used for:** Primary regression evaluation metric
   - **DOI:** 10.3354/cr030079

13. **Chai, T., & Draxler, R. R. (2014).** Root mean square error (RMSE) or mean absolute error (MAE)? Arguments against avoiding RMSE in the literature. *Geoscientific Model Development*, 7(3), 1247-1250.
   - **Metric:** Root Mean Squared Error (RMSE)
   - **Used for:** Secondary regression metric
   - **DOI:** 10.5194/gmd-7-1247-2014

14. **Nagelkerke, N. J. (1991).** A note on a general definition of the coefficient of determination. *Biometrika*, 78(3), 691-692.
   - **Metric:** R² (Coefficient of Determination)
   - **Used for:** Variance explained metric
   - **DOI:** 10.1093/biomet/78.3.691

### Classification Metrics

15. **Fawcett, T. (2006).** An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.
   - **Metric:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
   - **Used for:** Classification model discrimination ability
   - **DOI:** 10.1016/j.patrec.2005.10.010

16. **Davis, J., & Goadrich, M. (2006).** The relationship between Precision-Recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233-240).
   - **Metric:** PR-AUC (Precision-Recall Area Under Curve)
   - **Used for:** Primary classification metric (better for imbalanced classes)
   - **DOI:** 10.1145/1143844.1143874

17. **Sokolova, M., & Lapalme, G. (2009).** A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427-437.
   - **Metrics:** Precision, Recall, F1-Score, Accuracy
   - **Used for:** Comprehensive classification evaluation
   - **DOI:** 10.1016/j.ipm.2009.03.002

---

## Feature Engineering and Time-Series Analysis

18. **Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018).** Time series feature extraction on basis of scalable hypothesis tests (tsfresh–a python package). *Neurocomputing*, 307, 72-77.
   - **Concept:** Automated time-series feature engineering
   - **Used for:** Inspiration for temporal and sequence-based features
   - **DOI:** 10.1016/j.neucom.2018.03.067

19. **Hyndman, R. J., & Athanasopoulos, G. (2018).** *Forecasting: Principles and Practice* (2nd ed.). OTexts.
   - **Concept:** Time-series forecasting methods, lag features, rolling statistics
   - **Used for:** Feature engineering approach (lag features, rolling windows)
   - **URL:** https://otexts.com/fpp2/

20. **Bengio, Y., Simard, P., & Frasconi, P. (1994).** Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157-166.
   - **Concept:** Challenges in temporal sequence modeling
   - **Relevance:** Justification for using recent history (last 3 episodes + 7-day window)
   - **DOI:** 10.1109/72.279181

---

## Handling Imbalanced Data

21. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
   - **Concept:** Class imbalance handling (referenced for future work)
   - **Relevance:** High-intensity episodes (21.7%) vs low-intensity (78.3%)
   - **DOI:** 10.1613/jair.953

22. **He, H., & Garcia, E. A. (2009).** Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
   - **Concept:** Techniques for imbalanced classification
   - **Used for:** Understanding precision-recall trade-offs
   - **DOI:** 10.1109/TKDE.2008.239

---

## Medical and Clinical Context

### Tic Disorders

23. **American Psychiatric Association. (2013).** *Diagnostic and Statistical Manual of Mental Disorders* (5th ed.). Arlington, VA: American Psychiatric Publishing.
   - **Relevance:** Clinical definition and classification of tic disorders
   - **Used for:** Understanding intensity scales and episode characteristics
   - **ISBN:** 978-0890425558

24. **Leckman, J. F., Bloch, M. H., Smith, M. E., Larabi, D., & Hampson, M. (2010).** Neurobiological substrates of Tourette's disorder. *Journal of Child and Adolescent Psychopharmacology*, 20(4), 237-247.
   - **Relevance:** Understanding tic variability and temporal patterns
   - **DOI:** 10.1089/cap.2009.0118

25. **Conelea, C. A., & Woods, D. W. (2008).** The influence of contextual factors on tic expression in Tourette's syndrome: A review. *Journal of Psychosomatic Research*, 65(5), 487-496.
   - **Relevance:** Contextual factors (mood, triggers) in tic prediction
   - **Used for:** Feature selection rationale
   - **DOI:** 10.1016/j.jpsychores.2008.04.010

### Self-Reporting and Mobile Health

26. **Shiffman, S., Stone, A. A., & Hufford, M. R. (2008).** Ecological momentary assessment. *Annual Review of Clinical Psychology*, 4, 1-32.
   - **Relevance:** Self-reported episodic data collection methodology
   - **DOI:** 10.1146/annurev.clinpsy.3.022806.091415

27. **Kumar, S., Nilsen, W. J., Abernethy, A., Atienza, A., Patrick, K., Pavel, M., ... & Swendeman, D. (2013).** Mobile health technology evaluation: The mHealth evidence workshop. *American Journal of Preventive Medicine*, 45(2), 228-236.
   - **Relevance:** Mobile health data collection and analysis
   - **DOI:** 10.1016/j.amepre.2013.03.017

---

## Data Privacy and Ethics

28. **Voigt, P., & Von dem Bussche, A. (2017).** *The EU General Data Protection Regulation (GDPR)*. Springer International Publishing.
   - **Relevance:** Data privacy considerations for health data
   - **Used for:** Anonymization practices (user IDs only, no PII)
   - **DOI:** 10.1007/978-3-319-57959-7

29. **Office for Civil Rights, HHS. (2002).** Standards for privacy of individually identifiable health information. Final rule. *Federal Register*, 67(157), 53181-53273.
   - **Regulation:** HIPAA Privacy Rule
   - **Relevance:** Health data protection standards
   - **URL:** https://www.hhs.gov/hipaa/

---

## Statistical Methods

30. **Shapiro, S. S., & Wilk, M. B. (1965).** An analysis of variance test for normality (complete samples). *Biometrika*, 52(3/4), 591-611.
   - **Test:** Normality testing (used for understanding data distributions)
   - **DOI:** 10.2307/2333709

31. **Mann, H. B., & Whitney, D. R. (1947).** On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50-60.
   - **Test:** Non-parametric hypothesis testing
   - **Relevance:** Statistical significance of model improvements
   - **DOI:** 10.1214/aoms/1177730491

---

## Reproducibility and Best Practices

32. **Raschka, S. (2018).** Model evaluation, model selection, and algorithm selection in machine learning. *arXiv preprint arXiv:1811.12808*.
   - **Concept:** Best practices for model evaluation and selection
   - **Used for:** Train/test split methodology, cross-validation strategy
   - **URL:** https://arxiv.org/abs/1811.12808

33. **Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017).** The ML test score: A rubric for ML production readiness and technical debt reduction. *2017 IEEE International Conference on Big Data* (pp. 1123-1132).
   - **Concept:** ML system design and testing
   - **Used for:** Framework modularity and reproducibility
   - **DOI:** 10.1109/BigData.2017.8258038

34. **Peng, R. D. (2011).** Reproducible research in computational science. *Science*, 334(6060), 1226-1227.
   - **Concept:** Reproducibility standards
   - **Used for:** Fixed random seeds, version control, documented code
   - **DOI:** 10.1126/science.1213847

---

## Software Engineering and Development

35. **Git - Distributed Version Control System.** Software Freedom Conservancy.
   - **Tool:** Git (version control)
   - **Used for:** Code versioning, collaboration, reproducibility
   - **URL:** https://git-scm.com/

36. **Python Software Foundation. (2023).** Python Language Reference, version 3.10+.
   - **Language:** Python
   - **Used for:** All code implementation
   - **URL:** https://www.python.org/

37. **Jupyter Team. (2018).** Jupyter Notebooks - A publishing format for reproducible computational workflows. *Positioning and Power in Academic Publishing: Players, Agents and Agendas*, 87-90.
   - **Tool:** Jupyter Notebooks
   - **Used for:** Exploratory analysis and visualization
   - **DOI:** 10.3233/978-1-61499-649-1-87

---

## Visualization and Reporting

38. **Tufte, E. R. (2001).** *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.
   - **Concept:** Data visualization principles
   - **Used for:** Figure design and clarity
   - **ISBN:** 978-0961392147

39. **Few, S. (2012).** *Show Me the Numbers: Designing Tables and Graphs to Enlighten* (2nd ed.). Analytics Press.
   - **Concept:** Effective chart and graph design
   - **Used for:** Dashboard and performance visualization design
   - **ISBN:** 978-0970601971

---

## Related Work in Predictive Health Analytics

40. **Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., ... & Dean, J. (2019).** A guide to deep learning in healthcare. *Nature Medicine*, 25(1), 24-29.
   - **Concept:** Machine learning in healthcare applications
   - **Relevance:** Context for predictive health modeling
   - **DOI:** 10.1038/s41591-018-0316-z

41. **Rajkomar, A., Dean, J., & Kohane, I. (2019).** Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358.
   - **Concept:** Clinical applications of ML
   - **Relevance:** Predictive modeling for patient outcomes
   - **DOI:** 10.1056/NEJMra1814259

42. **Obermeyer, Z., & Emanuel, E. J. (2016).** Predicting the future—big data, machine learning, and clinical medicine. *New England Journal of Medicine*, 375(13), 1216-1219.
   - **Concept:** Time-series prediction in healthcare
   - **Relevance:** Predicting future health events from historical data
   - **DOI:** 10.1056/NEJMp1606181

---

## Dataset

43. **Project Dataset (2025).** Tic Episode Self-Report Data.
   - **Source:** Custom dataset collected via mobile application
   - **Size:** 1,533 episodes from 89 users
   - **Timeframe:** April 26 - October 25, 2025 (182 days)
   - **Features:** Timestamp, intensity (1-10), type, mood, trigger, description
   - **Format:** CSV with anonymized user IDs
   - **Repository:** https://github.com/aanishs/CSCI-461-Project

---

## Documentation and Technical References

44. **Scikit-learn User Guide. (2023).** Ensemble methods.
   - **URL:** https://scikit-learn.org/stable/modules/ensemble.html
   - **Accessed:** November 2025

45. **XGBoost Documentation. (2023).** XGBoost Parameters.
   - **URL:** https://xgboost.readthedocs.io/en/stable/parameter.html
   - **Accessed:** November 2025

46. **Pandas Documentation. (2023).** Time series / date functionality.
   - **URL:** https://pandas.pydata.org/docs/user_guide/timeseries.html
   - **Accessed:** November 2025

47. **Matplotlib Documentation. (2023).** Tutorials and examples.
   - **URL:** https://matplotlib.org/stable/tutorials/index.html
   - **Accessed:** November 2025

---

## Course Materials

48. **CSCI-461: Machine Learning Course Materials** (2025).
   - **Institution:** [Your Institution Name]
   - **Instructor:** [Instructor Name]
   - **Topics Covered:** Supervised learning, ensemble methods, model evaluation, hyperparameter tuning
   - **Semester:** Fall 2025

---

## Acknowledgments

### Tools and Infrastructure

- **GitHub:** Code hosting and version control
- **Claude Code (Anthropic):** Development assistance and code generation
- **ChatGPT (OpenAI):** Initial project brainstorming (if applicable)

### Computational Resources

- **Hardware:** Local development machine (macOS Darwin 22.6.0)
- **Python Environment:** Conda/pip virtual environment
- **IDE:** Jupyter Notebook, VS Code, Claude Code CLI

---

## Data Availability Statement

The anonymized dataset and complete code repository used in this study are available at:

**GitHub Repository:** https://github.com/aanishs/CSCI-461-Project

**Contents:**
- Raw data (CSV format with anonymized user IDs)
- Feature engineering pipeline
- Model training and evaluation scripts
- Hyperparameter search framework
- All figures and visualizations
- Complete documentation

**Reproducibility:** All results can be reproduced by running:
```bash
python run_hyperparameter_search.py --mode quick
```
with random seed 42 for deterministic results.

---

## Software Versions

For complete reproducibility, the following software versions were used:

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| scikit-learn | 1.3+ |
| xgboost | 2.0+ |
| lightgbm | 4.0+ |
| pandas | 2.0+ |
| numpy | 1.24+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| jupyter | 1.0+ |

Full dependency list available in `requirements.txt` in the repository.

---

## Citation Format

If you use this work, please cite as:

**APA Format:**
```
[Your Name]. (2025). Tic Episode Prediction: Hyperparameter Search for Predictive
Modeling of Tic Episode Patterns. CSCI-461 Machine Learning Course Project.
GitHub repository: https://github.com/aanishs/CSCI-461-Project
```

**BibTeX Format:**
```bibtex
@misc{tic_prediction_2025,
  author = {[Your Name]},
  title = {Tic Episode Prediction: Hyperparameter Search for Predictive Modeling
           of Tic Episode Patterns},
  year = {2025},
  publisher = {GitHub},
  journal = {CSCI-461 Machine Learning Course Project},
  howpublished = {\url{https://github.com/aanishs/CSCI-461-Project}},
  note = {Preliminary Report}
}
```

---

**Last Updated:** November 9, 2025
**Document Version:** 1.0
**Total References:** 48

---

## Additional Resources

### Related Projects
- **TicTrack App:** Mobile application for tic episode logging (hypothetical)
- **Healthcare ML Frameworks:** General frameworks for time-series health prediction

### Future Reading
For researchers extending this work, recommended reading includes:
- Deep learning for time-series (LSTM, Transformers)
- Personalized medicine and N-of-1 trials
- Causal inference in observational health data
- Multi-task learning for related predictions
