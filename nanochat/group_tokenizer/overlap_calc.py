"""
Calculate overlap design parameters.

Usage:
    uv run -m nanochat.group_tokenizer.overlap_calc--num_sub 8 --sub_per_final 2 --vocab_size 4096
    uv run -m nanochat.group_tokenizer.overlap_calc--num_sub 12 --sub_per_final 3 --vocab_size 4096
    uv run -m nanochat.group_tokenizer.overlap_calc--num_sub 16 --sub_per_final 4 --vocab_size 4096
"""
import argparse
from math import comb


def calc_overlap_design(num_sub: int, sub_per_final: int, vocab_size: int = 4096):
    """
    Calculate overlap design parameters for complete design (all combinations).

    Args:
        num_sub: number of sub-groups
        sub_per_final: sub-groups per final group
        vocab_size: total vocabulary size
    """
    # Tokens per sub-group
    tokens_per_sub = vocab_size / num_sub

    # Tokens per final group (noise level)
    tokens_per_final = tokens_per_sub * sub_per_final

    # Complete design: all combinations
    num_final = comb(num_sub, sub_per_final)

    # Each sub-group appears in how many finals?
    # It's paired with C(num_sub-1, sub_per_final-1) combinations of the remaining sub-groups
    overlap_k = comb(num_sub - 1, sub_per_final - 1)

    # Verify: num_sub * overlap_k = num_final * sub_per_final
    total_slots = num_final * sub_per_final
    total_appearances = num_sub * overlap_k
    assert total_slots == total_appearances, f"Math error: {total_slots} != {total_appearances}"

    return {
        "num_sub": num_sub,
        "sub_per_final": sub_per_final,
        "vocab_size": vocab_size,
        "tokens_per_sub": tokens_per_sub,
        "tokens_per_final": tokens_per_final,
        "num_final": num_final,
        "overlap_k": overlap_k,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate overlap design parameters")
    parser.add_argument("--num_sub", type=int, default=None, help="Number of sub-groups")
    parser.add_argument("--sub_per_final", type=int, default=None, help="Sub-groups per final group")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size")
    parser.add_argument("--target_noise", type=int, default=None,
                        help="Target noise level (tokens per final). If set, will search for valid configs.")
    args = parser.parse_args()

    if args.target_noise:
        # Search mode: find configs that achieve target noise level
        print(f"Searching for configs with tokens_per_final = {args.target_noise}, vocab_size = {args.vocab_size}")
        print()
        print(f"{'num_sub':>8} {'sub_per_final':>14} {'num_final':>10} {'overlap_k':>10} {'tokens/sub':>12} {'tokens/final':>13}")
        print("-" * 75)

        for num_sub in range(2, 65):
            for sub_per_final in range(1, num_sub):
                tokens_per_sub = args.vocab_size / num_sub
                tokens_per_final = tokens_per_sub * sub_per_final

                if abs(tokens_per_final - args.target_noise) < 0.01:
                    result = calc_overlap_design(num_sub, sub_per_final, args.vocab_size)
                    print(f"{result['num_sub']:>8} {result['sub_per_final']:>14} {result['num_final']:>10} "
                          f"{result['overlap_k']:>10} {result['tokens_per_sub']:>12.1f} {result['tokens_per_final']:>13.1f}")
    else:
        # Single calculation mode
        if args.num_sub is None or args.sub_per_final is None:
            parser.error("--num_sub and --sub_per_final required unless using --target_noise")
        result = calc_overlap_design(args.num_sub, args.sub_per_final, args.vocab_size)

        print(f"Overlap Design Parameters")
        print(f"=" * 40)
        print(f"Input:")
        print(f"  num_sub        = {result['num_sub']}")
        print(f"  sub_per_final  = {result['sub_per_final']}")
        print(f"  vocab_size     = {result['vocab_size']}")
        print()
        print(f"Derived:")
        print(f"  tokens_per_sub   = {result['tokens_per_sub']:.1f}")
        print(f"  tokens_per_final = {result['tokens_per_final']:.1f}  (noise level)")
        print(f"  num_final        = C({result['num_sub']},{result['sub_per_final']}) = {result['num_final']}")
        print(f"  overlap_k        = C({result['num_sub']-1},{result['sub_per_final']-1}) = {result['overlap_k']}")
        print()
        print(f"Summary: {result['num_final']} final groups, each token in {result['overlap_k']} groups")


if __name__ == "__main__":
    main()
