import os

import numpy as np
import torch
from pytorch_transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from utils import AbstractDataset, IntroDataset, choose_from_top, generate_intro

if torch.cuda.is_available():
    device = "cuda"
    print("using GPU")
else:
    device = "cpu"
    print("using CPU")


models_folder = "./trained_models/"
data_dir = "./clean_dataset/"

BEST_ABS_MODEL_PATH = "ABS_MODEL_lr_0.001_27_loss_947.pt"
BEST_INTRO_MODEL_PATH = "INTRO_MODEL_lr_0.0001_5_loss_4914.pt"
SAMPLES_DIR = "./samples_vlad/"
SAMPLES_NUM = 10


# Utils:
MINI_TRAIN = False




torch.cuda.empty_cache()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


if not os.path.exists(SAMPLES_DIR):
    os.mkdir(SAMPLES_DIR)


dataset = AbstractDataset(data_dir)
abstract_loader = DataLoader(dataset, batch_size=1, shuffle=True)
print("Loaded Abstract Dataset")
    
abstract_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
abs_model_path = os.path.join(models_folder, f"{BEST_ABS_MODEL_PATH}")
abstract_model.load_state_dict(torch.load(abs_model_path))
abstract_model.eval()
print("Loaded Abstract model")

intro_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
intro_model_path = os.path.join(models_folder, f"{BEST_INTRO_MODEL_PATH}")
intro_model.load_state_dict(torch.load(intro_model_path))
intro_model.eval()
print("Loaded Intro model")


generate_intro(
    abstract_loader,
    abstract_model,
    intro_model,
    SAMPLES_DIR,
    max_seq_len=500,
    num_samples=SAMPLES_NUM,
)
