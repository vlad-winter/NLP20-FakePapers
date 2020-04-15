# FakePapers_NLP_Project

## Task Definition
In order to make our University proud, we have decided that as all CS researches we preffer that someone else will do the job of writing papers for us. To accomplish this task we thought of a model, that according to our abstract idea, will write the rest of paper accordingly. So, formally speaking, giving an abstract, it would generate the rest of the paper by relevant sections, strating with the INTRO.

## Data description
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
Our architecture is based over gpt-2 model. At first stage, after data preprocessing, we finetune the pretrained gpt-2 model over our extracted abstracts.

Next, we use another pre-trained gpt-2 and finetune it in the following manner:
- for each abstract, intro pair we first do forward pass of the abstract on the previous model we learned.
- then, we take the past hidden state generated from that model, sent it as the past argument to the intro model and do forward pass on our given intro.
- backpropage the gradient loss.

* This process was tested on the intro, but can be performed on every section of the paper.



# Results

## Examples

## Conclusions

Unfortunately, as for the seen future, we will need to write our own papers. Nevertheless, the results were better then we expected and the generated introductions showed some correlation to the given abstract topic. Hopefully, someday we will be able to use this algorithm to defeat the Chinese.

## How to install

```bash 
pip install requirements.txt 
```

## Data Preprocessing

This stage required couple of stages 

## How to train

### Training the Abstract Model 
Run the following script:

```bash
python train_abs.py
```
### Training the Intro Model
Run the following script:

```bash
python train_intro.py
```
