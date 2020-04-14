from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm, trange
import torch
import os
import torch.nn.functional as F
from pytorch_transformers import (
    GPT2Tokenizer,
)


if torch.cuda.is_available():
    device = "cuda"
    print("using GPU")
else:
    device = "cpu"
    print("using CPU")


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def generate_intro(
    abs_dataloader,
    abs_model,
    intro_model,
    dir_name,
    max_seq_len,
    num_samples,
    verbose=False,
):
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    intro_num = 0
    abs
    with torch.no_grad():

        pbar = tqdm(iter(abs_dataloader), leave=False, total=num_samples)

        for idx, abstract in enumerate(pbar):
            print("#" * 30 + f"Sample #{idx+1}" + "#" * 30 + "\n")

            tokens_abs = tokenizer.encode(abstract[0])
            abstract_tens = torch.tensor(tokens_abs).unsqueeze(0).to(device)
            outputs = abs_model(abstract_tens, labels=abstract_tens, past=None)
            past = outputs[2]
            # trimmed_past = [p[:, :, :, -max_seq_len:, :] for p in past]
            intro_finished = False

            cur_ids = (
                torch.tensor(tokenizer.encode("<|startoftext|>"))
                .unsqueeze(0)
                .to(device)
            )

            for i in trange(max_seq_len):
                outputs = intro_model(cur_ids, labels=cur_ids, past=past)
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0, -1], dim=0)
                # Take the first(from only one in this case) batch
                # and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to("cpu").numpy(), n=n)
                # Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat(
                    [cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
                    dim=1,
                )
                # Add the last word to the running sequence

                if next_token_id in tokenizer.encode("<|endoftext|>"):
                    intro_finished = True
                    break

            if intro_finished or i == (max_seq_len - 1):
                intro_num = intro_num + 1


                output_list = list(cur_ids.squeeze().to("cpu").numpy())
                output_text = tokenizer.decode(output_list)

                if verbose:
                    print(abstract[0])
                    print(output_text)

                with open(f"{dir_name}/Sample #{idx+1}.txt", "w") as f:
                    f.write(abstract[0])
                    f.write("\n \n \n ")
                    f.write(output_text)


def sample_sequence(
    model,
    length,
    start_token=None,
    batch_size=None,
    context=None,
    temperature=1,
    top_k=0,
    device="cuda",
    sample=True,
):
    if start_token is None:
        assert context is not None, "Specify exactly one of start_token and context!"
        context = (
            torch.tensor(context, device=device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
    else:
        assert context is None, "Specify exactly one of start_token and context!"
        context = torch.full(
            (batch_size, 1), start_token, device=device, dtype=torch.long
        )
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


class AbstractDataset(Dataset):
    def __init__(self, data_dir, mini_train=False):
        super().__init__()

        self.data_dir = data_dir
        self.list_dir = os.listdir(data_dir)
        if mini_train:
            self.list_dir = self.list_dir[: int(len(self.list_dir) / 10)]

        self.start_of_text_token = "<|startoftext|>"
        self.end_of_text_token = "<|endoftext|>"

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, item):
        paper_num = self.list_dir[item]
        text_path = os.path.join(self.data_dir, paper_num, "abs.txt")
        f = open(text_path, encoding="utf-8")
        abstract_text = f.read()
        abstract_text = (
            f"{self.start_of_text_token} {abstract_text} {self.end_of_text_token}"
        )
        return abstract_text


class IntroDataset(Dataset):
    def __init__(self, data_dir, mini_train=False):
        super().__init__()

        self.data_dir = data_dir
        self.list_dir = os.listdir(data_dir)
        if mini_train:
            self.list_dir = self.list_dir[: int(len(self.list_dir) / 10)]

        self.start_of_text_token = "<|startoftext|>"
        self.end_of_text_token = "<|endoftext|>"

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, item):
        paper_num = self.list_dir[item]
        abstract_path = os.path.join(self.data_dir, paper_num, "abs.txt")

        f = open(abstract_path, encoding="utf-8")
        abstract_text = f.read()
        abstract_text = (
            f"{self.start_of_text_token} {abstract_text} {self.end_of_text_token}"
        )

        intro_path = os.path.join(self.data_dir, paper_num, "intro.txt")
        f = open(intro_path, encoding="utf-8")
        intro_text = f.read()
        intro_text = f"{self.start_of_text_token} {intro_text} {self.end_of_text_token}"
        return abstract_text, intro_text

    from torch.utils.data import DataLoader, Dataset


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits
    )
