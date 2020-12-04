ARG BASE_CONTAINER=balazska/dadteam:milestone1
FROM $BASE_CONTAINER

# Change working directory
#WORKDIR "/dadteam"

#RUN mkdir data
#RUN wget -c https://smresearchstorage.blob.core.windows.net/smcalflow-public/smcalflow.full.data.tgz -O - | tar -xz -C ./data
#RUN pip install git+https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis.git

# Copy package.json and package-lock.json
COPY ./DaDTeam_milestone_I.ipynb ./
COPY ./DaDTeam_milestone_II.ipynb ./
RUN mkdir glove
COPY ./glove/glove.6B.100d.txt ./glove
COPY ./lispress_tokenizer.py ./
RUN mkdir models
COPY ./models/model20201121_newtokenizer_50-50-1M.h5 ./models

USER root
RUN chown jovyan:users ./DaDTeam_milestone_I.ipynb
RUN chown jovyan:users ./DaDTeam_milestone_II.ipynb

