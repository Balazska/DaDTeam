# <center> We are the DaDTeam! </center>
<center> Balazs Fodor (GU87AO), Szilvia Hodvogner (W0VGDG), Gergely Dobreff (RZ3RVX) </center>

> The final solution can be found at the end of this readme [here](#final-solution)

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
Requirements: Python 3+ and tensorflow is assumed.

## Milestone I.
The [notebook](./DaDTeam_milestone_I.ipynb) and the data is included in this repository, and it contains codes to download and install the necessary components and libraries. (It also contains code to download the data, but it is only necessary if only the notebook will be downloaded).

## Milestone II.
Our goal for Milestone II. was to understand the problem and the dataset, and to create and train a generally well-performing model. To this end, we used the **seq2seq** model and aimed to train the model on the request-reply set of dialogues, thus we do not use the context of the dialogue yet.

The [notebook](./DaDTeam_milestone_II.ipynb) including all the codes is located in [this repository](https://github.com/Balazska/DaDTeam/blob/main/DaDTeam_milestone_II.ipynb).
### Preprocessing 

Before training, we selected all the question-and-answer sentence pairs in the data set, where the question is a command or request made by the user (e.g. What day of the week is tomorrow?). And the answer is the program code required to answer the question, called Lispress (e.g. (Yield: output (: dayOfWeek (Tomorrow)))). Based on Lispress, the system would generate a human-readable response (e.g. Tomorrow is Wednesday.), however this is not used.

Both the question and the answer are tokenized. The [Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) class of tensorflow was used to tokenize the questions. However, to tokenize Lispress, we used the tokenizer used by the competition announcers due to the syntactic characteristics of the language. After tokenization, *\<bos\>* and *\<eos\>* tokens were added to the beginning and end of the lispress.
    
### Training
#### Training set
Since we are utilizing the Seq2Seq model which contains an Encoder and a Decoder the training set for our Model is not obvious.
For training the input for the Encoder, the input for the Decoder and the output for the Decoder must be given. Therefore our trainingset looks like the following, where the *input sentence* is the user's request and the *current program* is the lispress for the input sentence, and *MAX_LEN_DEC* denotes the length of the decoder input and output.

Num | Encoder input | Decoder input | Decoder output
--- | --- | --- | ---
1 | input sequence 1 | current program 1 first token | current program 1 second token
2 | input sequence 1 | current program 1 first two tokens | current program 1 2. and 3. tokens
... | ... | ... | ...
n | input sequence 1 | current program 1 n-MAX_LEN_DEC-1 -> n-1 tokens | current program 1 n-MAX_LEN_DEC -> n tokens
n+1 | input sequence 2 | ... | ...

#### Modeling
We utilized a simple sequence to sequence model with LSTM encoder and decoder. The architecture of this model is shown in the following image (credit: [Akira Takezawa](https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639))
![Seq2Seq architecture](https://miro.medium.com/max/1250/1*LYGO4IxqUYftFdAccg5fVQ.png)

As we can see after the tokenized input is given to the Input layer of the decoder/encoder an Embedding layer takes each token (word) and converts them to a fixed sized vector. For embedding we used the **GLoVe.6B.100d** which was trained on 6 billion tokens and has a 400K big vocabulary and each token is represented by a 100 long vector. These vectors store the context and semantics of each token.

Summary of our Seq2Seq model is the following:
![Model summary](imgs/summary_milestone2.png?raw=true)


### Evaluating
In the SMCalFlow challenge, they compute the absolute accuracy, which means every word of the result must be right to get points. We also calculate our model's goodness like this, counting only a prediction true positive if all of the program tokens were predicted right. 

After the tokenization, our training set size turned out to be close to 5 million samples. Considering our best machine and its configuration, the training on the whole dataset takes approximately 55 hours. Before doing that, we did a smaller training on the fifth of it and then evaluated it to see if it is working fine. The accuracy of this test training was 11%. The model got the smaller codes right, but it had issues with more complex programs. It is not great, but we expected it to be something like this due to the dataset reduction and training of only three epochs.

After the first successful training, we re-run it on the whole training dataset. We initiated the batch size to 128 and 10 epochs. Currently (updated at 11.22.), the training is still running. It prints out the loss and accuracy values for both the train and the validation set, and the results look very promising. After only the first few epochs, the validation accuracy is exceeding 60% percent.

### Future plans
We also started to learn about new ways of improving the model. We are currently experimenting with Bidirectional LSTM layers. Bidirectional LSTMs are an extension of LSTMs. They are trained on the input sequence as-is and also on the reversed copy of the input sequence. Using BLSTM layers, we can provide additional context to the network, so we hope our model will be more precise. Besides, the Glove embedding is mainly for natural speaking, and our goal is to predict program codes, so we searched for alternative solutions. We found a method called the Attention mechanism, which is a layer that highlights the relevant features of the input data dynamically. We are currently experimenting with its implementation.

## Final solution:
> the final report can be read [here](DL_NHF_2020__DaDTeam.pdf)

> the final notebook can be accessed [here](DaDTeam_milestone_final.ipynb) 

### Abstract
The main challenge in conversational AI is to maintain the state of the conversation. To address this issue, Microsoft announced a challenge whose goal is to create such methods. We investigated the popular LSTM-based sequence to sequence model, and its extension with Attention and Bidirectional LSTM layers and evaluated its accuracy and the effect of different hyperparameters. The goal of the model was to generate a so called lispress program that represents the state of the conversational agent. The main difficulty is that every predicted token for a given input has to be correct in order for the prediction to be correct. Our model's accuracy could reach around 50-55%, and we are getting closer to submit our first solution to the challenge.

### Running the final Notebook
Google Colab is recommended to run the Notebook.
1. Upload the lispress_tokenizer.py file from the repository to the Colab's working directory
2. In the cell below the heading Glove, the appropriate rows should be uncommented. Since Colab does not have Glove downloaded, comment everything, but keep uncommented the lines below "GLOVE DOWNLOAD (IF NECESSARY)". Run the cell and the cell below (DOWNLOAD DATA), which downloads the SMCalFlow dataset.
3. Set the control variables in the next cell (Notebook control variables). For the meaning of the variables see Sec. 3.4. in the final report.
    - MODEL_TYPE can be either MODEL_BILSTM or MODEL_LSTM
    - RUN_MODE can be either RUN_TESTMODE or RUN_FULLMODE 
    - ATTENTION_LAYER can be True or False
    - HIDDEN_DIM, BATCH_SIZE, EPOCHS can be an arbitrary integer
    - PARAMETER_REPLACEMENT can be True or False
4. Run every cell and wait :)
