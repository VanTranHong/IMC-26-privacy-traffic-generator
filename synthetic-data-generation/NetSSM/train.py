# Code modified from: https://github.com/havenhq/mamba-chat/blob/main/train_mamba.py

import os
import glob
import torch
import argparse
import transformers

from datasets import load_dataset, concatenate_datasets, Dataset

from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer

from mamba_ssm.utils.hf import load_config_hf
from models.config_mamba import MambaConfig
from models.mixer_seq_simple import MambaLMHeadModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    def save_model(self, output_dir, num_epochs=None, epoch_save=False, _internal_call=None):
        if epoch_save:
          save_dir = output_dir + "/" + str(num_epochs) + "_epochs"
          self.model.save_pretrained(save_dir)
        else:
          self.model.save_pretrained(output_dir)

def get_checkpoint(model):
    cwd = os.getcwd()
    directory = os.path.join(cwd, model)
    ckpts = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for c in ckpts:
        if "epoch" in c:
            ckpts.remove(c)
    if not ckpts:
        return None
    load_ckpt = max(ckpts, key=lambda d: os.path.getmtime(os.path.join(directory, d)))
    ckpt = model + "/" + load_ckpt
    return ckpt

def get_dataset(data_path, tokenizer):
    if '.jsonl' in data_path:
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.map(lambda e: tokenizer(e['Data'], padding="longest"), batched=True)
        dataset = dataset.with_format("torch")
    else:
        find_arrow_files = lambda directory: glob.glob(os.path.join(directory, '*.arrow'))
        arrow_files = find_arrow_files(data_path)
        arrow_files.sort()
        dataset = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])
    return dataset
    
def parse_dtype(s):
    try:
        return getattr(torch, s)
    except AttributeError:
        raise argparse.ArgumentTypeError(f"Invalid dtype: {s}")

def run(args):
    transformers.logging.set_verbosity_info()
    resume = False
    try:
        load_ckpt = get_checkpoint(args.output)
        model = MambaLMHeadModel.from_pretrained(load_ckpt, dtype=args.torch_dtype, device="cuda")
        resume = True
    except:
        config_data = load_config_hf(args.model)
        config = MambaConfig(**config_data)
        model = MambaLMHeadModel(config, dtype=args.torch_dtype, device="cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = get_dataset(args.data_path, tokenizer)

    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output,
            save_total_limit=2,
            logging_steps=50,
            save_steps=500,
            num_train_epochs=args.num_epochs,
            # max_steps=args.num_steps,
        ),
    )
    if resume:
        trainer.train(get_checkpoint(args.output))
    else:
        trainer.train()

    trainer.save_model(args.output, args.num_epochs, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/configs/default")
    parser.add_argument("--output", type=str, default="checkpoints/nprint_os_0")
    parser.add_argument("--tokenizer", type=str, default="tokenizers/mia_netssm_nprint_os_tok")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="/net/scratch2/noise-lab/gen-net-attack/shadow-datasets/actual_data/nprint/netssm/in/nprint_os_10-pkt_0.jsonl")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--torch_dtype", type=parse_dtype, default="bfloat16", help="Torch dtype (e.g., float32, float16, bfloat16)")
    args = parser.parse_args()
    run(args)
