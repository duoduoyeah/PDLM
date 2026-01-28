

\section{Introduction}
% Introduce current state of the art, block diffusion models
 

What we dont understand here is that, why the model could predict a token, that is not the immediately next position, but the next position after skipping several positions.

\subsection{Experiment A}
Experiment A: still normal gpt2, with a jargon named ``target\_shift'', when this is 1, it means we to predict the token at position i+1 given tokens from position 0 to i; when this is 2, it means we predict the token at position i+2 given tokens from position 0 to i; 

By this experiment, we want to show that when the model predict a token that is far away, it actually perform worse than predicting the immediately next token. Be cautious here, is that when the model predict the token at position i+k, it just dont have the information of the tokens at position i+1 to i+k-1. 

We wonder if we add a supposed mask token to these positions(from i+1 to i+k-1), will the performance be improved? 


\textbf{Perspective 1:} A block diffusion model with block size $B$ is essentially a composite model containing $B$ different prediction capabilities. For example, when $B=4$, the model learns to predict tokens at four different relative positions. Specifically, for predicting position $i+k$ (where $k \in \{1,2,3,4\}$), the model is conditioned on:
\begin{itemize}
    \item Clean tokens (i.e., already-decoded tokens) from position $0$ to $i$
    \item A mixture of mask tokens and clean tokens at positions $i+1$ to $i+4$, depending on the diffusion timestep
\end{itemize}
The key distinction from standard autoregressive models is that positions $i+1$ to $i+k-1$ are not empty—they are explicitly filled with either mask tokens (indicating ``unknown'') or clean tokens (if already decoded in previous diffusion steps).


\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\linewidth]{figure/nextk}
    \caption{Block diffusion sampling procedure.}
    \label{fig:next-k}
\end{figure}

\subsection{Experiment B}
Experiment B: this time we will use block diffusion model with block size $B$, but for every time we validate this model, we just let it predict the token at position $i+k$, where $k$ is from $1$ to $B$. 

the key point here is that, when we compute loss, we will only compute the loss from position $i+k$ and this will be achieved by a special attn mask.

By this experiment, we want to show that when the block diffusion model is trained to predict a specific position, will it perform better than the normal GPT-2 with $target\_shift=k$? ``our hypo here is that there is no difference at all''

Here when validation, within that block, we will use mask token to fill the other positions except position i+k. Since we want to compare with the normal gpt2 with target\_shift=k, which means the model has no information about the tokens from position i+1 to i+k-1, so we use mask token to fill these positions, to make sure the model has no information about these positions.


Perspective 2: the mask token itself, is learned by the model as a one to all vocab mapping, since the mask token could be every token in the vocab.

\subsection{Experiment C}
Experiment C: this time, what we want to find out is that, is there a way, we could make the prediction of i+k, given the token from position 0 to i, perform as good as the prediction of i+1 given the token from position 0 to i? Our idea is that, we could add some extra information to help the model to do this. The extra information we used here is, instead of the give mask tokens, we give the model another special tokens named ``group tokens'', these 

Design A: the so called ``group tokens'' are actually still an mapping from 1 to vocab, but the difference is that , it is not mapping to all the vocab, but just a subset of the vocab. For example, if the vocab size is 4096, we could divide the vocab into 4 groups, each group contains 1024 tokens, then the group token 1 could represent the first 1024 tokens in the vocab, group token 2 could represent the second 1024 tokens in the vocab, and so on. By this way, when the model is predicting the token at position i+k, it could know that the token should be from which group, thus reduce the difficulty of the prediction.

We want to know in experiment C that:
1. such group tokens could help the model to improve the performance of predicting i+k given tokens from position 0 to i and group tokens from position i+1 to i+k?
2. is there a degree of group tokens that could make the performance of predicting i+k model close to predicting i+1?

What does degree of group token mean here? It means noisy degree, when the token is 1to1, it is clean token, every token in normal llm is clean token, when a group token is 1toN, this token we deemed it is noisy token, because it could represent N different tokens in the vocab. And when the N is large, the noisy degree is high.


\textbf{Decomposition} of this problem is to devide the decoding of a block to two stages: the first stage is from mask token to group token, the second stage is from group token to clean token. 

Since in Experiment C, we have shown that the group token could help the model to improve the performance of predicting i+k. However, we still do not know how to make the mask token transite to the group token during decoding. Thus in Experiment D, we will explore some possible methods to make this transite happen during decoding.

\subsection{Experiment D}
Experiment D: if the group token could help the model, then we want to propose a new model design, that is named parallel Denoising language model, in this model, when decoding, it will decode more than one position, but at each position, it will first start from decoding the mask token, and then transite to decoding the group token, when the decoding block size is larger than the needed transite forward step, then this model will has a larger tokens per step throughput than normal llm. Also, the block diffsuion model also do not has a large tokens per step throughput, and this is originally presumed by some people but this is not true, because it always need multiple steps to decode a block.

How to define a successful transite? I guess the metric should be like, if the new group token contain the target token in its group, then we deem this transite is successful.

Extra talking points:
1. why ar predict next several tokens failed? analyze the paper that add a head  at the end of arch to make the model predict more than one token at each step.

\subsection{The first Stage by MTP}
Experiment E is about, to test if MTP could achieve same accuracy with the bd3lm.
Multi-Token Prediction
