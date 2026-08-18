import argparse
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

def create_tokenizer(args):
    vocab = {str(i): i for i in range(256)}
    special_tokens = ["<|endoftext|>", "<|padding|>", "<|pkt|>", "<|unk|>"]
    if args.special_tokens is not None:
        special_tokens.extend(["<|" + tok + "|>" for tok in args.special_tokens.split()])

    vocab["<|unk|>"] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(special_tokens)

    netssm_tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|endoftext|>", pad_token="<|padding|>", unk_token="<|unk|>", clean_up_tokenization_spaces=True)
    netssm_tok.save_pretrained(args.tokenizer_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a simple tokenizer mapping [0, 255] to single tokens.")
    parser.add_argument("--special_tokens", type=str, help="Special tokens to include in the tokenizer, input as space-delimitted string (e.g., \"netflix facebook amazon\").")
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("--tokenizer_name", type=str, required=True, help="Name to save the tokenizer.")
    args = parser.parse_args()

    create_tokenizer(args)
