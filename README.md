# Transitioning-from-Fingerstick-to-Continuous-Glucose-Monitoring-Predictions
Semester Project - Data Driven Diabetes Management - BME University of Bern 2023
Authors: Daniel Kerber, Diego Zeiter & Julian Piller 

## Abstract 
Diabetes, a chronic metabolic condition that leads to high blood glucose levels, affects over 540 million adults worldwide and is underscored by 6.7 million annual related deaths. Continuous glucose monitoring (CGM) devices are typically used in people with type 1 diabetes (PwT1D), with readings every 1-15 minutes. People with type 2 diabetes (PwT2D), which represent over 90\% of the total diabetic population, rely on blood glucose information taken a few times a day from self-monitoring blood glucose (SMBG) devices. Our study aims to bridge this gap in diabetes management technology, particularly for PwT2D, who primarily use SMBG samples. We hypothesize that it is feasible to develop an AI-based Long Short-Term Memory (LSTM) model that can accurately predict CGM profiles from intermitting SMBG readings and other relevant features. In addition, our objective is to minimize the number of required SMBG inputs while ensuring a good capability to predict glucose fluctuations safely. We used the OhioT1DM dataset, which contains data from PwT1D, including CGM, SMBG, basal and bolus insulin, and additional explanatory variables. After implementing the LSTM model, which included data preprocessing, adding artificial SMBG values, and determining the appropriate prediction window size, we performed three experiments to understand the impact of the different model variables. We determined no clear performance difference between using a specific patient or a generalized model. Additionally, we confirmed the link between lower SMBG sampling frequency and a corresponding decrease in CGM prediction accuracy (1h: RMSE of 19.50; 8h: RMSE of 51.50). Furthermore, the ideal prediction window was between 5 and 8.33 hours. Lastly, we showed that an LSTM model can accurately predict CGM profiles from intermitting SMBG readings and other relevant features. Future research should focus on model fine-tuning to ensure safe glucose predictions, consider different preprocessing approaches, test the model using a dataset containing PwT2D, and verify the impact of using artificial SMBGs versus only real SMBGs.



![SMBG_to_CGM_schematic](https://github.com/pillerjulian/Transitioning-from-Fingerstick-to-Continuous-Glucose-Monitoring-Predictions/assets/125559438/f8e91fda-1c59-4662-ae4e-04754974d10f)



## Run the program
### Local Machine
To run the code, an conda environment with the needed packages must first be created. <br />
**conda env create --name your_env_name --file=environment.yml** <br />
Activate the created environment and open the jupyter notebook or python file from this environment. 

### Ubelix
- Change the default mail address to your own mail address inside 'script.sh' from the folder 'scripts' to receive notifications about your job progress.
- Transfer the files 'create_ubelix_env.sh', 'requirements.txt', 'script.sh' from the folder 'scripts' into Ubelix in a folder called diabetes.
- Transfer to Ubelix the 'dataset' folder (NOT ADDED IN THIS REPOSITORY, FOR MORE INFORMATION CHECK THIS LINK: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) with the subfolders 'Ohio2018_processed' and 'Ohio2020_processed' into the diabetes folder.
- Transfer the python file 'src/main.py' into Ubelix in a new folder called 'src' in diabetes.
- Make sure that you have execution rights for main.py (Check using 'ls -la' and in case you dont have the right add it using 'chmod +x main.py')
- The following point will create a new conda environment called diabetesProject with the correct Python version and the packages from 'requirements.txt'
- We recommend to execute manually the following commands in Ubelix instead of running the 'create_ubelix_env.sh' file:
  - module load Anaconda3
  - eval "$(conda shell.bash hook)"
  - conda create --name diabetesProject python=3.10.13
  - conda activate diabetesProject
  - pip install -r requirements.txt (make sure you are in the same directory as 'requirements.txt')
- Now create a folder 'results' inside diabetes. This folder will hold the results of 'main.py'.
- Now you can run the 'script.sh' to train the model and obtain the results.
- After completion of the job, adapt and run 'fetch_results.sh' on your local machine to download the results from Ubelix or download them manually.
