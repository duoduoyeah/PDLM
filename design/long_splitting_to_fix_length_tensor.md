# Hierarchical Tokenization

## 1. Structural Splitting

Split text at punctuation and whitespace boundaries where cross-boundary dependency is weak. These are purely symbol-based, no word-level heuristics:

| Boundary | Strength | Example |
|---|---|---|
| `\n\n+` | Strongest (paragraph) | Topic/scene shift |
| `[.!?]+\s+` | Sentence end | New statement |
| `[""]` | Quote boundary | Narrator vs. character voice |
| `[;:]\s+` | Strong clause break | Independent clauses |
| `—` | Em dash | Interruption/aside |
| `,\s+` | Weakest (clause) | Dependent clause |

This is analogous to how BPE uses a regex (GPT-4 pattern) to split text into token-level chunks — but one level up. BPE chunks are ~1 word; these segments are ~1 sentence/clause.

## 2. Segment Encoder/Decoder

Encoder maps variable-length segments (1~K tokens) into fixed-shape latent, decoder reconstructs. Trained together with reconstruction loss (like CV autoencoders for latent diffusion).

**Note:** The goal is not compression — it's to express variable-length input as a fixed-length representation. A 5-token segment and a 15-token segment both become `[K, hidden]`. This uniformity enables the main LLM to treat segments as its atomic units.

```
1~K tokens  →  Encoder  →  [K, hidden] + length signal  →  Decoder  →  1~K tokens
                           (always fixed)     ↑                ↑
                                              └────────────────┘
```

The encoder outputs both the latent representation and a length signal. The decoder uses both to reconstruct the original segment. This is self-contained — the fixed-shape output contains everything needed to reconstruct, including length.

## 3. Two-Level Architecture

The fixed-shape segment representations become the input sequence to the main LLM, attending causally to all previous segments:

```
Text:  "He wanted to put socks on every cloud!" | "Then the clouds will be my silly army!" | "he laughed."
        |                                         |                                         |
        Segment Encoder                           Segment Encoder                           Segment Encoder
        |                                         |                                         |
        S1 [K, hidden]                            S2 [K, hidden]                            S3 [K, hidden]
        |                                         |                                         |
        LLM input pos 1                           LLM input pos 2                           LLM input pos 3
                                                  attends to S1                             attends to S1, S2
```

Instead of the LLM seeing 26 individual tokens with each attending to all previous tokens, it sees 3 segment representations with each attending to all previous segments.

**Core insight:** Segments replace tokens as the fundamental unit of the main model. Structural boundaries (punctuation, paragraph breaks) become the "tokenization" at this higher level — just like BPE regex splits define boundaries at the token level.

This buys a much longer effective context window because attention operates over segments instead of individual tokens, and cross-boundary dependencies are inherently weaker, so compressing within a segment loses less important information.
