#!/bin/bash

# Define variables
REMOTE_USER="dk22k072"
REMOTE_HOST="submit03.unibe.ch"
REMOTE_FOLDER= "/storage/homefs/dk22k072/diabetes/src"
LOCAL_DESTINATION="C:/UniBern/DataDrivenDiabetes/Transitioning-from-Fingerstick-to-Continuous-Glucose-Monitoring-Predictions/results"
RESULT_FILE_NAME="test1.png"
LOSS_FILE_NAME= "loss1.png"

# Download the zip file to the local machine
scp ${REMOTE_USER}@${REMOTE_HOST}:${RESULT_FILE_NAME} ${LOCAL_DESTINATION}
scp ${REMOTE_USER}@${REMOTE_HOST}:${LOSS_FILE_NAME} ${LOCAL_DESTINATION}

# Optional: Remove the zip file from the remote server
ssh ${REMOTE_USER}@${REMOTE_HOST} "rm ${RESULT_FILE_NAME}"
ssh ${REMOTE_USER}@${REMOTE_HOST} "rm ${LOSS_FILE_NAME}"