ARG BASE_CONTAINER=balazska/dadteam:milestone1
FROM $BASE_CONTAINER

# Change working directory
#WORKDIR "/dadteam"

#RUN mkdir data
#RUN wget -c https://smresearchstorage.blob.core.windows.net/smcalflow-public/smcalflow.full.data.tgz -O - | tar -xz -C ./data
#RUN pip install git+https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis.git

# Copy package.json and package-lock.json
COPY ./DaDTeam_milestone_I.ipynb ./

