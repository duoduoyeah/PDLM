"""
Explain entropy PPL as effective token count.

Entropy PPL = exp(H), where H = -sum(p(x) * log(p(x))) is the entropy
of the model's output distribution. It equals the number of tokens in
a uniform distribution with the same entropy — i.e., the "effective
number of choices" the model sees.

Usage:
    uv run -m utils.explain_entropy_ppl --entropy_ppl 2.30
    uv run -m utils.explain_entropy_ppl --entropy_ppl 2.30 --noise_count 128 --vocab_size 4096
"""

import argparse
import math


def explain(entropy_ppl, vocab_size=4096, noise_count=None):
    entropy = math.log(entropy_ppl)
    max_entropy = math.log(vocab_size)

    print(f"entropy_ppl = {entropy_ppl:.2f}")
    print(f"  = exp(H), H = {entropy:.4f} nats")
    print(f"  = model effectively choosing from ~{entropy_ppl:.1f} tokens")
    print(f"  = 1/{entropy_ppl:.1f} avg probability on the top candidate")
    print()
    print(f"  vocab_size = {vocab_size}")
    print(f"  max entropy_ppl (uniform over vocab) = {vocab_size}")
    print(f"  entropy ratio H/H_max = {entropy/max_entropy:.4f}")

    if noise_count is not None:
        noise_entropy = math.log(noise_count)
        print()
        print(f"  noise_count = {noise_count}")
        print(f"  entropy_ppl if uniform over noise set = {noise_count}")
        print(f"  narrowing: {noise_count} noise tokens -> ~{entropy_ppl:.1f} effective choices")
        print(f"  information gained from context = {noise_entropy - entropy:.4f} nats = {(noise_entropy - entropy)/math.log(2):.2f} bits")


def main():
    parser = argparse.ArgumentParser(description="Explain entropy PPL as effective token count")
    parser.add_argument("--entropy_ppl", type=float, required=True)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--noise_count", type=int, default=None)
    args = parser.parse_args()

    explain(args.entropy_ppl, args.vocab_size, args.noise_count)


if __name__ == "__main__":
    main()
