import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    WarmupLinearSchedule,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
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

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 500
GRAD_ACC_STEPS = 2
ABS_MODEL_NAME = "ABS_MODEL"
MAX_GRAD_NORM = 1
RUN_TAG = f"lr_{LEARNING_RATE}"
GEN_SAMPLES_NUMS = 2
SAVE_EVERY = 5


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


# Abstract Learning

dataset = AbstractDataset(data_dir, MINI_TRAIN)
abstract_loader = DataLoader(dataset, batch_size=1, shuffle=True)


abstract_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
abstract_model.train()

optimizer = AdamW(abstract_model.parameters(), lr=LEARNING_RATE)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS, t_total=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
losses = []

tmp_abstract_tens = None

if not os.path.exists(models_folder):
    os.mkdir(models_folder)

print("=" * 30 + f"Training Abstract Model" + "=" * 30)

for epoch in range(EPOCHS):

    print("=" * 30 + f"EPOCH {epoch+1}/{EPOCHS} started" + "=" * 30)

    pbar = tqdm(iter(abstract_loader), leave=False, total=len(abstract_loader))

    for idx, abstract in enumerate(pbar):

        # "Fit as many abstract sequences into
        # MAX_SEQ_LEN sequence as possible" logic start
        abstract_tens = (
            torch.tensor(tokenizer.encode(abstract[0])).unsqueeze(0).to(device)
        )
        # Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if abstract_tens.size()[1] > MAX_SEQ_LEN:
            continue

        # The first abstract sequence in the sequence
        if not torch.is_tensor(tmp_abstract_tens):
            tmp_abstract_tens = abstract_tens
            continue
        else:
            # The next abstract does not fit in so we process the
            # sequence and leave the last abstract as the start for next sequence
            if tmp_abstract_tens.size()[1] + abstract_tens.size()[1] > MAX_SEQ_LEN:
                work_abstracts_tens = tmp_abstract_tens
                tmp_abstract_tens = abstract_tens
            else:
                # Add the abstract to sequence, continue and try to add more
                tmp_abstract_tens = torch.cat(
                    [tmp_abstract_tens, abstract_tens[:, 1:]], dim=1
                )
                continue

        # Sequence ready, process it trough the model

        outputs = abstract_model(work_abstracts_tens, labels=work_abstracts_tens)
        loss, logits = outputs[:2]
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
            abstract_model.state_dict(),
            os.path.join(
                models_folder,
                f"{ABS_MODEL_NAME}_{RUN_TAG}_{epoch}_loss_{int(sum_loss)}.pt",
            ),
        )

    print(f"sum loss {sum_loss}")
    losses.append(sum_loss)
    batch_count = 0
    sum_loss = 0.0

    print("=" * 30 + "Starting Sampling:" + "=" * 30)
    # Sample
    text = "We present here"

    context_tokens = tokenizer.encode(text)
    out = sample_sequence(
        model=abstract_model,
        length=MAX_SEQ_LEN,
        context=context_tokens,
        start_token=None,
        batch_size=GEN_SAMPLES_NUMS,
    )
    generated = 0
    out = out[:, len(context_tokens) :].tolist()
    for i in range(GEN_SAMPLES_NUMS):
        generated += 1
        text = tokenizer.decode(out[i])
        print(text)
        print
        ("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)



    plt.figure()
    plt.plot(np.array(losses))
    plt.title(f"Abstract Model {RUN_TAG} Train loss:")
    plt.savefig(f"./logs/Abs_model_{RUN_TAG}.png",)
