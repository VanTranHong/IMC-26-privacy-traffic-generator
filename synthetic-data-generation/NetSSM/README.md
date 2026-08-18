# NetSSM

## 📌 Attribution

This folder vendors **NetSSM**, developed by researchers at the [Noise Lab](https://github.com/noise-lab):

- 💻 Original repository: [github.com/noise-lab/netssm](https://github.com/noise-lab/netssm) (commit `bc5ce67`, `git archive`'d unmodified — no local edits)
- 📜 License: MIT (per the upstream repository)

All credit for the method and implementation goes to the original authors. NetSSM builds on [Mamba/Mamba-2](https://github.com/state-spaces/mamba) as its underlying state-space model architecture (see `models/`), and uses Hugging Face `transformers`/`datasets` for tokenization and training.

### What's vendored here vs. what you provide
This folder contains the full NetSSM source (preprocessor, tokenizer scripts, `models/`, `train.py`, `generation/`) plus its small bundled example (`example/input/*.pcap`, the default model config, and the pre-built `nm_tokenizer_multi_netflix` tokenizer) — everything needed to run the pipeline below. Not included (must be obtained separately, see the relevant steps):
- Go toolchain and Python/CUDA environment (install steps below).
- `causal-conv1d` / `mamba-ssm` compiled wheels — version-and-GPU-specific, install per your CUDA setup.
- The pre-trained `netflix_multi_100k_30_epochs` checkpoint (downloaded via `gdown`, see [Generation](#generation)) and any of your own training data/checkpoints — these are large, run-specific artifacts, not part of the source repo.

---

> 📓 **Prefer a hands-on walkthrough?** [`example/train_netssm_from_scratch.ipynb`](./example/train_netssm_from_scratch.ipynb) runs this entire pipeline end-to-end (preprocessing → tokenizing → training from scratch → resuming from a checkpoint → generation → PCAP conversion) on the four sample video-streaming PCAPs in [`example/input/`](./example/input/), and is also [open-able directly in Colab](https://colab.research.google.com/github/noise-lab/netssm/blob/main/example/train_netssm_from_scratch.ipynb).

## Requirements

  * Go >= v1.21.4
  * Python3 >= 3.10
  * libpcap

## Setup

Install dependencies: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install "causal-conv1d==1.5.0" "mamba-ssm==2.2.4"`

## Dataset preparation

Raw PCAPs must be pre-processed into a string representation sequence of packets and then tokenized, for use with NetSSM.

### Data pre-processing

Setup the preprocessor:
```bash
cd preprocessing
go mod init netssm_preprocessor
go mod tidy
go build
```

This builds the preprocessor binary called `netssm_preprocessor`. The preprocessor has the following usage options:

```
Usage of ./netssm_preprocessor:
  -in-csv string
        CSV file with PCAP paths (default "./")
  -in-dir string
        Directory with PCAPs (default "./")
  -label string
        Blanket label to use, if all PCAPs are of same service/type
  -label-csv in-dir
        CSV mapping pcaps in in-dir to their corresp. label (default "./")
  -out string
        Output JSONL dataset name (default "./")
  -truncate int
        OPTIONAL -- Length to truncate all samples to (default -1)
```

One of `-in-csv` or `-in-dir` must be provided. Similarly, either `label` or `-label-csv` must be provided. If `-in-csv` is used, the CSV pointed to should have header `File`, where the value in this column should be the full path to each PCAP. The CSV file for `-label-csv` should have the header `File,Label`, where the value in `File` should be only the filename of each PCAP. See `example/input/labels.csv` for an example of the labels file.

The preprocessor will parse raw PCAP files into a string of the raw bytes of comprising each packet in a capture. This string is prepended with the traffic label/type, and each packet is delimited by a `<|pkt|>` special token. For example: `<|netflix|> 226 70 154 78 108 47... <|pkt|> 00 128 153 192 8... <|pkt|>`.

### Tokenizers

NetSSM's tokenizer maps the raw byte decimal values in $[0, 255]$ 1:1 to corresponding tokens. Both training from scratch or fine-tuning on new data will likely require a new custom tokenizer to handle this data. Use `tokenizers/create_tokenizer.py` to create this.

```bash
python3 create_tokenizer.py [-h] [--special_tokens SPECIAL_TOKENS] --tokenizer_name TOKENIZER_NAME

optional arguments:
  -h, --help            show this help message and exit
  --special_tokens SPECIAL_TOKENS
                        Special tokens to include in the tokenizer, input as space-delimitted string (e.g., "netflix facebook amazon").

required arguments:
  --tokenizer_name TOKENIZER_NAME
                        Name to save the tokenizer.
```

The output of this script is a folder containing various files that comprise a custom tokenizer.

### Tokenizing Dataset

We tokenize the dataset produced by the pre-processing step before training, to more efficiently use GPU allocation. The script for this is at `preprocessing/create_tok_dataset.py`.

```bash
python3 create_tok_dataset.py [-h] [--batch_size BATCH_SIZE] [--num_proc NUM_PROC] [--max_len MAX_LEN] [--padding] --tokenizer TOKENIZER --data_path DATA_PATH --out_path OUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Number of inputs to process at a time.
  --num_proc NUM_PROC   Number of processes to spawn for tokenization.
  --max_len MAX_LEN     Length to truncate all sequences to (used to handle for memory constraints on GPU).
  --padding             If used, will pad inputs to the length of the longest sequence in the batch.

required arguments:
  --tokenizer TOKENIZER
                        Path to tokenizer.
  --data_path DATA_PATH
  --out_path OUT_PATH
```

The output of this script is a folder containing `.arrow` files of the tokenized representations of the dataset created in [Data pre-preprocessing](#data-pre-processing).

### Summary

Putting it all together, the following code block explains how to convert a directory of PCAPs to their tokenized representations, using the `-label-csv` option for label mapping:

```bash
# 1. Parse PCAPs to string representations
./netssm_preprocessor -in-dir <PATH_TO_PCAP_DIR> -label-csv <PATH_TO_LABEL_CSV> -out <PATH_TO_DATASET_JSONL>
# 2. Create a new tokenizer including traffic labels
python3 create_tokenizer.py --special_tokens "netflix facebook amazon" --tokenizer_name <CUSTOM_TOKENIZER_OUTPUT_PATH>
# 3. Apply tokenizer to the raw dataset, for use in model training
python3 create_tok_dataset.py --max_len 100000 --padding --tokenizer <CUSTOM_TOKENIZER_OUTPUT_PATH> --data_path <PATH_TO_DATASET_JSONL> --out_path <DATASET_OUT_PATH>
```

The resulting tokenized dataset at `<DATASET_OUT_PATH>` is now usable for training/fine-tuning.

## Training/Fine-tuning

⚠️ **NetSSM (and as source the original Mamba/2) currently does not support multi-GPU training out of the box. See this issue ([#84](https://github.com/state-spaces/mamba/issues/84)) for some potential workarounds (untested for this repo).** ⚠️

After creating a dataset and tokenizer, and tokenizing the dataset using the above steps, training can be run using the `train.py` script in the root directory:

```bash
python train.py \
  --model=<PATH_TO_MODEL_CONFIG>
  --output=<PATH_TO_OUTPUT_FOLDER> \
  --data_path=<DATASET_OUT_PATH> \
  --tokenizer=<CUSTOM_TOKENIZER_OUTPUT_PATH> \
  --num_epochs=<NUM_EPOCHS>
```

`<PATH_TO_MODEL_CONFIG>` should resemble a file similar to that at `checkpoints/configs/default`, which specifies the model parameters.

If a prior model training checkpoint exists at `<PATH_TO_OUTPUT_FOLDER>`, the training script will automatically load from this checkpoint, and resume training. Otherwise, the model will begin training from scratch.

There are additional training parameters that can be passed to `train.py`; see the arguments in the file itself.

### Fine-tuning

If you are using an existing model checkpoint, but want to fine-tune on new data with labels that do not exist in the checkpointed model/tokenizer, follow the steps in [Dataset preparation](#dataset-preparation), creating a new tokenizer that contains the new labels for the new data, and creating the tokenized dataset. Then when using `train.py`, set `--output` to the directory containing the starting checkpoint for fine-tuning, `--tokenizer` to the new tokenizer, and `--data_path` to the tokenized dataset.

## Generation

Generation runs in two steps: (1) generating the raw tokens corresponding to bytes in a PCAP and (2) converting these tokens to the actual trace.

### Example with a pre-trained checkpoint

We provide a toy pre-trained checkpoint that can be used for generation out of the box. Download the pre-trained checkpoint:

```bash
source venv/bin/activate
gdown 1koMbDyaTi0buF1eoDplqOFtJLX-ssS6a
mv netflix_multi_100k_30_epochs.zip ./checkpoints && cd ./checkpoints && unzip netflix_multi_100k_30_epochs.zip && mv checkpoint-176460 netflix_multi_100k_30_epochs && cd ..
```

Then follow the steps for (1) and (2) below.

### 1. Raw token generation

Example usage prompting to generate a PCAP of 1,000 packets of Netflix traffic using the pre-trained model checkpoint and corresponding multi-flow tokenizer.

```bash
python3 ./generation/generate.py \
  --prompt "<|netflix|>" \
  --model "./checkpoints/netflix_multi_100k_30_epochs" \
  --tokenizer "./tokenizers/nm_tokenizer_multi_netflix" \
  --genlen 1000
```

There are a number of generation parameters in this script that can be adjusted. Generation will write by default to `./inference/EXP_1/RUN_1/generated.txt`.

### 2. Convert raw tokens to PCAP

Convert the raw text generated output to a PCAP using `./generation/conversion.py`.

```bash
python3 ./generation/conversion.py <PATH/TO/GENERATED.TXT> <PATH/TO/OUTPUT/PCAP>
```

