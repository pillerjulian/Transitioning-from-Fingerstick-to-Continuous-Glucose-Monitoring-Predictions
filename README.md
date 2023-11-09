# Transitioning-from-Fingerstick-to-Continuous-Glucose-Monitoring-Predictions
Semester Project - Data Driven Diabetes Management 

## Environment for Jupyter Notebook on Local Machine
To run the code, an conda environment with the needed packages must first be created. \
**conda env create --name your_env_name --file=environment.yml** \ 
Activate the created environment and open the jupyter notebook from this environment. 

## Ubelix

- Transfer the files 'create_ubelix_env.sh', 'requirements.txt', 'script.sh' from the folder 'scripts' into Ubelix.
- Transfer to Ubelix the 'dataset' folder with the subfolders 'Ohio2018_processed' and 'Ohio2020_processed'. 
- Run 'create_ubelix_env.sh' in Ubelix. This will create a new environent with the correct Python version and the packages from "requirements.txt"
- Now you can run the 'script.sh' to train the model and obtain the results.

## Scope 
Diabetes is a chronic metabolic disease due to insufficient or the lack of insulin production from
pancreatic β-cells. Insulin is the primary regulator for the cellular metabolism of blood glucose (BG)
and any malfunction in its production results in elevated BG levels. For people with diabetes, it is
crucially important to avoid the onset of extreme hypo- and hyperglycemic events. To this end, a
plethora of statistical and machine learning (ML) algorithms [1]. have been introduced to support
People with Type 1 Diabetes (PwT1D) in managing the glucose metabolism by predicting future BG
levels and raising alarms [2]. Most of those predictive tools utilize continuous glucose monitoring
(CGM), demographic, and patient data. We assume that people with any type of diabetes could
benefit from predictions about future glycemic excursions. However, CGM is mostly used by PwT1D
and the majority of People with Type 2 Diabetes (PwT2D) use self-monitoring blood glucose
(SMBG) measurements such as finger prink samples. Therefore, the question arises, what
minimum number of finger stick blood samples are required to predict future blood glucose values
and adverse events as accurate as when CGM is used?

## Data 
You will be working with recorded data from 12 different individuals with T1D. The data was
released in the OhioT1DM dataset. You will have access to information such as CGM, SMBG,
basal insulin rate, bolus injection, the self-reported time and type of a meal, plus the patient’s
carbohydrate estimate for the meal and more. The measurements are provided at intervals of
minutes.

## Experiment 
Within the framework of this experiment, you will evaluate what number of finger stick blood
samples are required to generate a CGM-like glucose profile by augmenting regular finger stick
values (taken prior the BG before main meals and sleep) through artificially selecting CGM values
at certain time points as seeds to recreate the whole CGM sequence, experimenting for different
prediction horizons (I.e., 1h, 2h etc.).
