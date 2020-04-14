import logging
import os

import numpy as np
import torch
from pytorch_transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    WarmupLinearSchedule,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import (
    AbstractDataset,
    IntroDataset,
    choose_from_top,
    generate_intro,
    top_k_logits,
    sample_sequence
)

# Trains on 10% of the data:
MINI_TRAIN = False

# Hyper Params:

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-4
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 500
GRAD_ACC_STEPS = 2
INTRO_MODEL_NAME = "INTRO_MODEL"
MAX_GRAD_NORM = 1
RUN_TAG = f"lr_{LEARNING_RATE}"
SAVE_EVERY = 5
abs_path = "./trained_models/ABS_MODEL_lr_0.001_27_loss_947.pt"


if torch.cuda.is_available():
    device = "cuda"
    print("using GPU")
else:
    device = "cpu"
    print("using CPU")


models_folder = "./trained_models"
data_dir = "./clean_dataset"


# Utils:




torch.cuda.empty_cache()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


"""# Intro Learning"""


intro_dataset = IntroDataset(data_dir, MINI_TRAIN)
intro_loader = DataLoader(intro_dataset, batch_size=1, shuffle=True)
print(f"Loaded Intro Dataset, length: {len(intro_dataset)}")

logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(
    logging.ERROR
)  # No warning on sample size

intro_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
intro_model.train()

abstract_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
abstract_model.load_state_dict(torch.load(abs_path))
abstract_model.eval()

optimizer = AdamW(intro_model.parameters(), lr=LEARNING_RATE)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS, t_total=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
losses = []

torch.cuda.empty_cache()

tmp_intro_tens = None

print("=" * 30 + f"Training Intro Model" + "=" * 30)

for epoch in range(EPOCHS):

    print("=" * 30 + f"EPOCH {epoch+1}/{EPOCHS} started" + "=" * 30)

    pbar = tqdm(iter(intro_loader), leave=False, total=len(intro_loader))

    for idx, paper in enumerate(pbar):

        abs_txt = paper[0][0]
        int_txt = paper[1][0]

        tokens_abs = tokenizer.encode(abs_txt)
        tokens_int = tokenizer.encode(int_txt)

        lst = []
        tmp_in = tokens_int
        while len(tmp_in) > MAX_SEQ_LEN:
            lst += [tmp_in[:MAX_SEQ_LEN]]
            tmp_in = tmp_in[-MAX_SEQ_LEN:]
        lst += [tmp_in]

        if len(tokens_abs) > MAX_SEQ_LEN:
            continue

        abstract_tens = torch.tensor(tokens_abs).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = abstract_model(abstract_tens, labels=abstract_tens, past=None)
        past = outputs[2]

        # past = None
        loss = 0
        for curr_tokens in lst:
            work_intros_tens = torch.tensor(curr_tokens).unsqueeze(0).to(device)

            trimmed_past = [p[:, :, :, -500:, :] for p in past]
            outputs = intro_model(
                work_intros_tens, labels=work_intros_tens, past=trimmed_past
            )
            loss_curr, logits = outputs[:2]
            past = outputs[2]
            loss += loss_curr

        loss = loss / GRAD_ACC_STEPS
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count % GRAD_ACC_STEPS == 0:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            abstract_model.zero_grad()

    # Store the model after each epoch to compare the performance of them
    if epoch % SAVE_EVERY == 0:
        torch.save(
            intro_model.state_dict(),
            os.path.join(
                models_folder,
                f"{INTRO_MODEL_NAME}_{RUN_TAG}_{epoch}_loss_{int(sum_loss)}.pt",
            ),
        )

    print(f"sum loss {sum_loss}")
    losses.append(sum_loss)
    batch_count = 0
    sum_loss = 0.0

    plt.figure()
    plt.plot(np.array(losses))
    plt.title(f"Intro Model {RUN_TAG} Train loss:")
    plt.savefig(f"./logs/Intro_model_{RUN_TAG}.png",)
