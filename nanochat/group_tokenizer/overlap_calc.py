"""
Calculate overlap design parameters.

Usage:
    # All combos mode (complete design):
    uv run -m nanochat.group_tokenizer.overlap_calc --mode all_combos --num_sub 8 --sub_per_final 2 --vocab_size 4096

    # Flexible mode:
    uv run -m nanochat.group_tokenizer.overlap_calc --mode flexible --num_sub 128 --sub_per_final 2 --overlap_k 4 --vocab_size 4096

    # Search for configs at target noise level:
    uv run -m nanochat.group_tokenizer.overlap_calc --target_noise 1024 --vocab_size 4096
"""
import argparse
from math import comb


def calc_all_combos_design(num_sub: int, sub_per_final: int, vocab_size: int = 4096):
    """
    Calculate overlap design parameters for all_combos mode (all combinations).

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
        "mode": "all_combos",
        "num_sub": num_sub,
        "sub_per_final": sub_per_final,
        "vocab_size": vocab_size,
        "tokens_per_sub": tokens_per_sub,
        "tokens_per_final": tokens_per_final,
        "num_final": num_final,
        "overlap_k": overlap_k,
    }


def calc_flexible_design(num_sub: int, sub_per_final: int, overlap_k: int, vocab_size: int = 4096):
    """
    Calculate overlap design parameters for flexible mode.

    Args:
        num_sub: number of sub-groups
        sub_per_final: sub-groups per final group
        overlap_k: number of final groups each token belongs to
        vocab_size: total vocabulary size
    """
    # Tokens per sub-group
    tokens_per_sub = vocab_size / num_sub

    # Tokens per final group (noise level)
    tokens_per_final = tokens_per_sub * sub_per_final

    # num_final = (num_sub * overlap_k) / sub_per_final
    # This must be an integer
    total_sub_slots = num_sub * overlap_k
    assert total_sub_slots % sub_per_final == 0, (
        f"Invalid config: num_sub * overlap_k ({total_sub_slots}) "
        f"must be divisible by sub_per_final ({sub_per_final})"
    )
    num_final = total_sub_slots // sub_per_final

    return {
        "mode": "flexible",
        "num_sub": num_sub,
        "sub_per_final": sub_per_final,
        "vocab_size": vocab_size,
        "tokens_per_sub": tokens_per_sub,
        "tokens_per_final": tokens_per_final,
        "num_final": num_final,
        "overlap_k": overlap_k,
    }


def print_result(result):
    """Pretty print a design result."""
    print(f"Overlap Design Parameters ({result['mode']} mode)")
    print(f"=" * 50)
    print(f"Input:")
    print(f"  num_sub        = {result['num_sub']}")
    print(f"  sub_per_final  = {result['sub_per_final']}")
    if result['mode'] == 'flexible':
        print(f"  overlap_k      = {result['overlap_k']} (specified)")
    print(f"  vocab_size     = {result['vocab_size']}")
    print()
    print(f"Derived:")
    print(f"  tokens_per_sub   = {result['tokens_per_sub']:.1f}")
    print(f"  tokens_per_final = {result['tokens_per_final']:.1f}  (noise level)")
    if result['mode'] == 'all_combos':
        print(f"  num_final        = C({result['num_sub']},{result['sub_per_final']}) = {result['num_final']}")
        print(f"  overlap_k        = C({result['num_sub']-1},{result['sub_per_final']-1}) = {result['overlap_k']}")
    else:
        print(f"  num_final        = {result['num_sub']} * {result['overlap_k']} / {result['sub_per_final']} = {result['num_final']}")
    print()
    print(f"Summary: {result['num_final']} final groups, each token in {result['overlap_k']} groups")
    print(f"Config name: n{int(result['tokens_per_final'])}_k{result['overlap_k']}_g{result['num_final']}")


def main():
    parser = argparse.ArgumentParser(description="Calculate overlap design parameters")
    parser.add_argument("--mode", type=str, choices=["all_combos", "flexible"], default=None,
                        help="Design mode")
    parser.add_argument("--num_sub", type=int, default=None, help="Number of sub-groups")
    parser.add_argument("--sub_per_final", type=int, default=None, help="Sub-groups per final group")
    parser.add_argument("--overlap_k", type=int, default=None, help="Overlap k (flexible mode only)")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size")
    parser.add_argument("--target_noise", type=int, default=None,
                        help="Target noise level (tokens per final). If set, will search for valid configs.")
    args = parser.parse_args()

    if args.target_noise:
        # Search mode: find configs that achieve target noise level
        print(f"Searching for configs with tokens_per_final = {args.target_noise}, vocab_size = {args.vocab_size}")
        print()
        print("All Combos Mode (complete designs):")
        print(f"{'num_sub':>8} {'sub_per_final':>14} {'num_final':>10} {'overlap_k':>10} {'tokens/sub':>12} {'tokens/final':>13}")
        print("-" * 75)

        found_any = False
        for num_sub in range(2, 257):
            for sub_per_final in range(1, min(num_sub, 5)):
                tokens_per_sub = args.vocab_size / num_sub
                tokens_per_final = tokens_per_sub * sub_per_final

                if abs(tokens_per_final - args.target_noise) < 0.01:
                    result = calc_all_combos_design(num_sub, sub_per_final, args.vocab_size)
                    print(f"{result['num_sub']:>8} {result['sub_per_final']:>14} {result['num_final']:>10} "
                          f"{result['overlap_k']:>10} {result['tokens_per_sub']:>12.1f} {result['tokens_per_final']:>13.1f}")
                    found_any = True

        if not found_any:
            print("  (no configurations found)")

        print()
        print("Flexible Mode examples (custom overlap_k):")
        print(f"{'num_sub':>8} {'sub_per_final':>14} {'overlap_k':>10} {'num_final':>10} {'tokens/sub':>12} {'tokens/final':>13}")
        print("-" * 75)

        # Show some flexible mode examples for target noise
        for num_sub in [64, 128, 256, 512]:
            for sub_per_final in [1, 2]:
                tokens_per_sub = args.vocab_size / num_sub
                tokens_per_final = tokens_per_sub * sub_per_final

                if abs(tokens_per_final - args.target_noise) < 0.01:
                    for overlap_k in [2, 4, 8]:
                        if (num_sub * overlap_k) % sub_per_final == 0:
                            result = calc_flexible_design(num_sub, sub_per_final, overlap_k, args.vocab_size)
                            print(f"{result['num_sub']:>8} {result['sub_per_final']:>14} {result['overlap_k']:>10} "
                                  f"{result['num_final']:>10} {result['tokens_per_sub']:>12.1f} {result['tokens_per_final']:>13.1f}")
    else:
        # Single calculation mode
        if args.mode is None:
            args.mode = "all_combos"

        if args.num_sub is None or args.sub_per_final is None:
            parser.error("--num_sub and --sub_per_final required unless using --target_noise")

        if args.mode == "flexible" and args.overlap_k is None:
            parser.error("--overlap_k required for flexible mode")

        if args.mode == "all_combos":
            result = calc_all_combos_design(args.num_sub, args.sub_per_final, args.vocab_size)
        else:
            result = calc_flexible_design(args.num_sub, args.sub_per_final, args.overlap_k, args.vocab_size)

        print_result(result)


if __name__ == "__main__":
    main()
