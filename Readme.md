# <center> We are the DaDTeam! </center>
<center> Balazs Fodor, Szilvia Hodvogner, Gergely Dobreff </center>

## Goal
We choose topic NLP2 (ChatBot based on deep learning) for the Deep Learning Course's homework. Our goal is to build a Neural Network for the [SMCalFlow challenge](https://microsoft.github.io/task_oriented_dialogue_as_dataflow_synthesis/). This competition was announced by Microsoft Semantic Machine, the motivation for this competition is that one of the central challenges in conversational AI is the design of a dialogue state representation that agents can use to reason about the information and actions available to them. They have developed a new representational framework for dialogue that enables efficient machine learning of complex conversations.

## Dataset
SMCalFlow is a large English-language dialogue dataset, featuring natural conversations about tasks involving calendars, weather, places, and people. It has 41,517 conversations annotated with dataflow programs. In contrast to existing dialogue datasets, this dialogue collection was not based on pre-specified scripts, and participants were not restricted in terms of what they could ask for and how they should accomplish their tasks. As a result, SMCalFlow is qualitatively different from existing dialogue datasets, featuring explicit discussion about agent capabilities, multi-turn error recovery, and complex goals.

## Reference
The original scientific paper about the framework, the dataset and their baseline solution can be found [here](https://www.mitpressjournals.org/doi/10.1162/tacl_a_00333), a shorter summary in the form of a blog post can be found [at this link](https://www.microsoft.com/en-us/research/blog/dialogue-as-dataflow-a-new-approach-to-conversational-ai/). 

Microsoft also opensourced a [github repository](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis) containing codes to reproduce their results in the article. The library can be installed with pip using the following command:
```sh
pip install git+https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis.git
```

## Target of the challenge
The main target of the challenge is to create algorithms that are able to maintain and update conversation states stored as a program (or as a graph). The output for each user message should be a so called lispress string which is a representation of the stored agent program (see examples above).

The predictions can be evaluated in [codalab](https://codalab.org/), using Microsoft's predefined scripts. Predictions need to be made for all the validation data, and submitted to the codalab platform, and the scripts will evaluate the accuracy. 


## Roadmap
Since the original article contains a baseline solution for this challenge, our first step will be to understand and reproduce the baseline solution to better understand the problem the article addresses.

After that, with the deep analysis of the state of the art solutions, we will extend, or replace the baseline solution with our custom algorithm.

Our final goal is to create a model whose prediction outperforms the baseline solution.

## Running the codes
The [notebook](./DaDTeam_milestone_I.ipynb) and the data is included in this repository, and it contains codes to download and install the necessary components and libraries. (It also contains code to download the data, but it is only necessary if only the notebook will be downloaded).

Python 3+ and tensorflow is assumed.

### Docker
There is a docker repository for this project, the docker image containing the notebook for milestone 1 can be started with the following command:
```
docker run -p 8888:8888 --name dadteam_milestone1 balazska/dadteam:milestone1_v1
```
After that, the jupyter notebook server can be reached at the http://localhost:8888/?token=TOKEN url, the full url will be displayed in the terminal window. 


## Milestore II.
