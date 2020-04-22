# FakePapers_NLP_Project

## Table of Contents
1. [Task Definition](#Task-Definition)
2. [Data Description](#Data-Description)
3. [Data Preprocessing](#Data-Preprocessing)
4. [Architecture](#Architecture)

## Task Definition
In order to make our University proud, we have decided that as all CS researches we preffer that someone else will do the job of writing papers for us. To accomplish this task we thought of a model, that according to our abstract idea, will write the rest of paper accordingly. So, formally speaking, giving an abstract, it would generate the rest of the paper by relevant sections, strating with the INTRO.

## Data Description
The dataset was downloaded from arxiv using a python script scraper. We downloaded and preprocessed over 2,000 papers, s.t. for each paper the abstract and the matching introuction texts were extracted.

### Data Preprocessing

1. **Getting the papers**: in the `/data_prepocessing/data_download` folder, run the following script:
```bash
python complete_dataset.py
```
it will create a folder named `papers folder'

2. **Parsing LaTeX** and extracting relevant parts: 
```bash
python data_prepocessing/extract.py --path PATH_TO_PAPERS_FOLDER
```

## Architecture
Most Machine-Learning papers are divided to similar sections: introduction; previous work; preliminaries; method; results; conclusion, each written in a different tone and flavor. We've decided that each section should use a different model in order to capture those differences.  
In order to do that, we've designed the architecture based on hierarchy of gpt-2 models. First process the abstract, and then continue the processing with a different model for each section.

The learning process is as follows:
At the first stage, after data preprocessing, we fine-tune the pretrained gpt-2 model over our extracted abstracts.  
Next, we use another pre-trained gpt-2 model and fine-tune it in the following manner:
- for each <abstract,section> pair perform a forward pass of given abstract on the 'abstract' model previously learned.
- then, take the past hidden state generated from that model, and send it as the past argument to the section model and do forward pass on our given section.
- backpropage the gradient loss only for the section model (excluding the 'abstract' model)

## Results
This architecture was tested on the intro, but can be performed on every section of the paper.

### Examples
TBA

## Evaluating the model
We did not find an empirical way to evaluate the model, except of our sheer admiration.  
To evaluate the model we generated a bunch of into sections given abstracts, and were looking for the specific "intro tone" we know from papers we've read, and for a correlation between the topics talked about in the 'abstract' and the generated 'intro'.  
The model has performed above and beyond our expectations (which were quite low as you can probably tell).

## Conclusions
Unfortunately, as for the seen future, we will need to write our own papers.  
Nevertheless, the results were better then we expected and the generated introductions showed some correlation to the given abstract topic.

Future work can be done by using a better hierarchy, having a custome GPT-2 like model that receives an output from 'abstract' model and use it for each section model in a different way than using the 'past' argument.  


## Using the Code

### How to install

```bash 
pip install requirements.txt 
```

### Data preprocessing

This stage required couple of stages 

### How to train

#### Training the 'abstract' model 
Run the following script:

```bash
python train_abs.py
```
#### Training the 'intro' model
Run the following script:

```bash
python train_intro.py
```
