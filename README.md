# Fake Papers - NLP Project

## Table of Contents
- [Task Definition](#Task-Definition)
- [Data Description](#Data-Description)
- [Preprocessing](#Preprocessing)
- [Architecture](#Architecture)
- [Results](#Results)
  - [Examples](#Examples)
- [Evaluation](#Evaluation)
- [Conclusions](#Conclusions)
- [Using the Code](#Using-the-Code)

## Task Definition
In order to make our University proud, we have decided that as all CS researches we preffer that someone else will do the job of writing papers for us. To accomplish this task we thought of a model, that according to our abstract idea, will write the rest of paper accordingly. So, formally speaking, giving an abstract, it would generate the rest of the paper by relevant sections, strating with the INTRO.

## Data Description
The dataset was downloaded from arxiv using a python script scraper. We downloaded and preprocessed over 2,000 papers, s.t. for each paper the abstract and the matching introuction texts were extracted.

## Preprocessing
TBA

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

#### Abstract [Li, G., & Yu, Y. (2016)](https://arxiv.org/abs/1609.02077):
Visual saliency is a fundamental problem in both cognitive and computational sciences, including computer vision. In this paper, we discover that a high-quality visual saliency model can be learned from multiscale features extracted using deep convolutional neural networks (CNNs), which have had many successes in visual recognition tasks. For learning such saliency models, we introduce a neural network architecture, which has fully connected layers on top of CNNs responsible for feature extraction at three different scales. We then propose a refinement method to enhance the spatial coherence of our saliency results. Finally, aggregating multiple saliency maps computed for different levels of image segmentation can further boost the performance, yielding saliency maps better than those generated from a single segmentation. To promote further research and evaluation of visual saliency models, we also construct a new large database of N challenging images and their pixelwise saliency annotations. Experimental results demonstrate that our proposed method is capable of achieving state-of-the-art performance on all public benchmarks, improving the F-Measure by N \% and N \% respectively on the MSRA-B dataset and our new dataset (HKU-IS), and lowering the mean absolute error by N \% and N \% respectively on these two datasets. 
 
#### Generated Introduction:
The availability of large amounts of data for visual understanding and understanding is critical for many tasks such as computer vision and natural language processing, such as image classification and segmentation. However, it is not always possible to train a deep convolutional neural network (CNN) on such large data. For example, a CNN trained on ImageNet _cite_ is not able to learn a saliency map of a single image, but it can learn a saliency from the input image, which is not always available. This problem has been extensively studied in the literature, and it has been addressed in several works, such as the ImageNet dataset _cite_. In the literature, the goal of this work is to develop a CNN model for image classification and segmentation, but this is not always feasible. For example, in the ImageNet dataset _cite_, a CNN trained on ImageNet is not able to learn an image-level saliency map of the same image. In this work, we explore how to train a CNN model to learn saliency from images, and we present the first attempt of this approach, namely the DeepCNN model, which is a deep convolutional neural network (ConvNets) trained on ImageNet _cite_. We use the ConvNets to train a CNN model for visual recognition tasks such as classification and detection. In this paper, we present an approach to improve the performance of the CNN in the ImageNet dataset _cite_, which contains over N million images and N million pixels. The proposed CNN model is based on a fully connected layer, which consists of three layers, one for each pixel, and the other two layers for each pixel. In the first layer, we use the convolutional layer to extract features from the image, and the convolutional layer is responsible for generating saliency map of the same image. In the second layer, we use the global convolutional layer to generate saliency map. The global convolutional layer is responsible for generating saliency map of a similar image. The global convolutional layer is responsible for generating saliency map of a similar image. The global convolutional layer is responsible for generating saliency map of a similar image. The global convolutional layer is responsible for generating saliency map of a similar image. The global convolutional layer is responsible for generating saliency map of a similar image.

## Evaluation
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


1. **Getting the papers**: in the `/data_prepocessing/data_download` folder, run the following script:
```bash
python complete_dataset.py
```
it will create a folder named `papers folder`

2. **Parsing LaTeX** and extracting relevant parts: 
```bash
python data_prepocessing/extract.py --path PATH_TO_PAPERS_FOLDER
```

### Training

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
