"""
Chat with a PDLM checkpoint using its block generation.

Example:
python -m scripts.chat_pdlm -b 8 --max-new-tokens 16 --dump True
python -m scripts.chat_pdlm -b 8 --max-new-tokens 128 --dump False
Mary likes toy,
Tom and Mary will go out today,
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
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
dump_index = 0

def _format_block(tokens):
    if torch.is_tensor(tokens):
        tokens = tokens.tolist()
    if isinstance(tokens, list) and len(tokens) == 1:
        return tokens[0]
    return tokens

def _decode_token(token_id):
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return repr(token_id)

def _decode_ids(token_ids):
    if isinstance(token_ids, list):
        return [_decode_ids(item) for item in token_ids]
    return _decode_token(token_ids)

def _round_probs(values, decimals=3):
    if isinstance(values, list):
        return [_round_probs(item, decimals=decimals) for item in values]
    if isinstance(values, float):
        return round(values, decimals)
    return values

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
        dump_index += 1
        dump_dir = os.path.join(os.getcwd(), "pdlm_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = os.path.join(dump_dir, f"pdlm_dump_{timestamp}_{dump_index:04d}.txt")
        dump_lines = []
        for block in block_debug:
            pure_topk_ids = _format_block(block["pure_ids"])
            pure_topk_probs = _round_probs(_format_block(block["pure_probs"]), decimals=3)
            pure_topk_text = _decode_ids(pure_topk_ids)

            max_ids = [ids[0] for ids in pure_topk_ids]
            max_probs = [probs[0] for probs in pure_topk_probs]
            max_texts = [texts[0] for texts in pure_topk_text]
            max_tokens_info = list(zip(max_texts, max_ids, max_probs))

            ids_line = f"[dump] step_{block['step']} pure_topk_ids={pure_topk_ids}"
            probs_line = f"[dump] step_{block['step']} pure_topk_probs={pure_topk_probs}"
            text_line = f"[dump] step_{block['step']} pure_topk_text={pure_topk_text!r}"
            max_line = f"[dump] step_{block['step']} max_tokens={max_tokens_info}"

            print(ids_line)
            print(probs_line)
            print(text_line)
            print(max_line)
            dump_lines.extend([ids_line, probs_line, text_line, max_line])
        if dump_lines:
            with open(dump_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(dump_lines) + "\n")
            print(f"[dump] wrote {dump_path}")

    print("\n", end="", flush=True)
    if response_tokens:
        print(tokenizer.decode(response_tokens), end="", flush=True)
    print()
    conversation_tokens.extend(response_tokens)

    if args.prompt:
        break
