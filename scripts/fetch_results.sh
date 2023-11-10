#!/bin/bash

# Define variables
REMOTE_USER="dk22k072"  # TODO: Change this to your username
REMOTE_HOST="submit03.unibe.ch"
BASE_REMOTE_FOLDER="/storage/homefs"
REMOTE_FOLDER="${BASE_REMOTE_FOLDER}/${REMOTE_USER}/diabetes/results"
LOCAL_DESTINATION="C:/UniBern/DataDrivenDiabetes/Transitioning-from-Fingerstick-to-Continuous-Glucose-Monitoring-Predictions/results" # TODO: Change this to your local destination

# Download the results directory to the local machine
scp -r ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_FOLDER} ${LOCAL_DESTINATION}

## Optional: Remove the contents of the results directory from the remote server
ssh ${REMOTE_USER}@${REMOTE_HOST} "rm -r ${REMOTE_FOLDER}/*"


