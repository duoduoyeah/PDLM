"""
Chat with a PDLM checkpoint using its block generation.

Example:
python -m scripts.chat_pdlm --model-tag d4 --step 000050
python -m scripts.chat_pdlm -b 2 --max-new-tokens 128 --dump False
Mary likes toy,
Tom and Mary will go out today,
"""
import argparse
from contextlib import nullcontext
import os

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, find_largest_model
from nanochat.tokenizer import get_tokenizer
from nanochat.pdlm import PDLM, PDLMConfig

parser = argparse.ArgumentParser(description="Chat with a PDLM model")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
parser.add_argument("-p", "--prompt", type=str, default="", help="Prompt the model, get a single response back")
parser.add_argument("--max-new-tokens", type=int, default=16, help="Max new tokens to generate")
parser.add_argument("-b", "--bucket-size", type=int, default=8, help="Bucket size for block generation")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type for eval")
parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--dump", type=str, default="True", choices=["True", "False"])
args = parser.parse_args()


def load_pdlm(checkpoint_dir, device, step=None, model_tag=None):
    if model_tag is None:
        model_tag = find_largest_model(checkpoint_dir)
    ckpt_dir = os.path.join(checkpoint_dir, model_tag)
    if step is None:
        step = find_last_step(ckpt_dir)
    model_data, _, meta_data = load_checkpoint(ckpt_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    model_config = PDLMConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = PDLM(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    return model, meta_data


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "base_checkpoints")
model, meta = load_pdlm(checkpoint_dir, device, step=args.step, model_tag=args.model_tag)
tokenizer = get_tokenizer()

bos = tokenizer.get_bos_token_id()

print("\nNanoChat PDLM Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]
dump_enabled = args.dump == "True"

def _format_block(tokens):
    if tokens.ndim == 2 and tokens.size(0) == 1:
        return tokens[0].tolist()
    return tokens.tolist()

def _decode_block(token_ids):
    # Best-effort decode for display; fall back to repr if decode fails.
    try:
        return tokenizer.decode(token_ids)
    except Exception:
        return repr(token_ids)

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if user_input.lower() == "clear":
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    conversation_tokens.extend(tokenizer.encode(user_input))

    print("\n[model] thinking...", flush=True)

    prompt_tokens = list(conversation_tokens)
    max_total_tokens = len(prompt_tokens) + args.max_new_tokens
    bucket_size = args.bucket_size
    if max_total_tokens % bucket_size != 0:
        max_total_tokens = ((max_total_tokens + bucket_size - 1) // bucket_size) * bucket_size

    with autocast_ctx:
        if dump_enabled:
            ids, block_debug = model.generate_with_blocks(
                prompt_tokens,
                max_total_tokens,
                bucket_size=bucket_size,
            )
        else:
            ids = model.generate(prompt_tokens, max_total_tokens, bucket_size=bucket_size)
    ids = ids[0].tolist()
    response_tokens = [tok for tok in ids[len(prompt_tokens):] if tok >= 0]

    if dump_enabled:
        for block in block_debug:
            noisy_block = _format_block(block["noisy_ids"])
            pure_block = _format_block(block["pure_ids"])
            noisy_text = _decode_block(noisy_block)
            pure_text = _decode_block(pure_block)
            print(f"[dump] step_{block['step']} noisy_block={noisy_block} pure_block={pure_block}")
            print(f"[dump] step_{block['step']} noisy_text={noisy_text!r}")
            print(f"[dump] step_{block['step']} pure_text={pure_text!r}")
            if "next_ids" in block:
                next_block = _format_block(block["next_ids"])
                next_text = _decode_block(next_block)
                print(f"[dump] step_{block['step']} next_block={next_block} next_text={next_text!r}")

    print("\n", end="", flush=True)
    if response_tokens:
        print(tokenizer.decode(response_tokens), end="", flush=True)
    print()
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break
