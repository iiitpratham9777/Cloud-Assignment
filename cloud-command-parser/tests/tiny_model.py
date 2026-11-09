"""Build a tiny, randomly initialised GPT-NeoX model + locally trained tokenizer (no downloads).

For pipeline tests only: it exercises training/train.py and service.model_loader.Generator with
the real PyTorch/Transformers code, but it is NOT Pythia-70M and its output quality is meaningless.
"""
import json
from pathlib import Path


def build_tiny_base_model(directory, corpus):
    """Write a tiny base model and tokenizer into ``directory`` and return the path."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=500, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)
    # Like Pythia's tokenizer: no token_type_ids (GPT-NeoX does not accept them in generate()).
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                                   model_input_names=["input_ids", "attention_mask"])
    fast.save_pretrained(directory)
    cfg = GPTNeoXConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, max_position_embeddings=256, bos_token_id=0, eos_token_id=0)
    GPTNeoXForCausalLM(cfg).save_pretrained(directory, safe_serialization=True)
    return directory


def write_config(path, base_model, train_file, output_dir, loss_csv, epochs=30, lr=3e-3, records=64):
    """Write a training config (JSON is valid YAML) pointing at the tiny model."""
    cfg = {
        "base_model": str(base_model), "train_file": str(train_file), "max_records": records,
        "learning_rate": lr, "batch_size": 8, "max_seq_length": 128, "output_dir": str(output_dir),
        "loss_csv": str(loss_csv), "threads": 2, "mask_prompt_in_loss": True,
        "versions": {"v1": {"seed": 42, "epochs": epochs}},
        "library_versions": {"python": "3.11", "torch": "2.4.1", "transformers": "4.44.2",
                             "tokenizers": "0.19.1", "numpy": "1.26.4"},
    }
    Path(path).write_text(json.dumps(cfg), encoding="utf-8")
    return path
