## 3 Background on NLMs & GPT-2

To generate a text sequence of length T, x = [x1, · · · , x<sup>T</sup> ], neural language models (NLM) [\(Mikolov et al.,](#page-10-11) [2010\)](#page-10-11) generate every token x<sup>t</sup> conditioned on the previous word tokens:

<span id="page-1-1"></span>
$$p(\boldsymbol{x}) = \prod_{t=1}^{T} p_{\boldsymbol{\theta}}(x_t | x_{< t}), \tag{1}$$

<span id="page-1-0"></span><sup>2</sup><https://github.com/ChunyuanLI/Optimus>

where  $x_{< t}$  indicates all tokens before t, and  $\theta$  is the model parameter. In NLMs, each one-step-ahead conditional in (1) is modeled by an expressive family of neural networks, and is typically trained via maximum likelihood estimate (MLE). Perhaps the most well-known NLM instance is GPT-2 (Radford et al., 2019), which employs Transformers (Vaswani et al., 2017) for each conditional, and  $\theta$  is learned on a huge amount of OpenWeb text corpus. GPT-2 has shown surprisingly realistic text generation results, and low perplexity on several benchmarks. GPT-3 (Brown et al., 2020) was recently proposed to further scale up NLMs to 175 billion parameters, showing impressive results on few-shot learning on multiple language tasks.

However, the only source of variation in NLMs, GPT2 and GPT3 is modeled in the conditionals at every step: the text generation process only depends on previous word tokens, and there is limited capacity for the generation to be guided by the higher-level structures that are likely presented in natural language, such as tense, topics or sentiment.

### 4 Pre-trained Latent Space Modeling

### <span id="page-2-3"></span>4.1 Pre-training Objectives

To facilitate high-level guidance in sentence generation, OPTIMUS organizes sentences in a universal latent (or semantic) space, via pre-training on large text corpora. Each sample in this space can be interpreted as outlines of the corresponding sentences, guiding the language generation process performed in the symbolic space (Subramanian et al., 2018). This naturally fits within the learning paradigm of latent variable models such as VAEs (Kingma and Welling, 2013; Bowman et al., 2016), where the latent representations capture the high-level semantics/patterns. It consists of two parts, generation and inference, enabling a bidirectional mapping between the latent space and symbolic space.

Generation The generative model (decoder) draws a latent vector z from the continuous latent space with prior p(z), and generates the text sequence x from a conditional distribution  $p_{\theta}(x|z)$ ; p(z) is typically assumed a multivariate Gaussian, and  $\theta$  represents the neural network parameters. The following auto-regressive decoding process is usually used:

$$p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}) = \prod_{t=1}^{T} p_{\boldsymbol{\theta}}(x_t|x_{< t}, \boldsymbol{z}).$$
 (2)

Intuitively, VAE provides a "hierachical" generation procedure:  $z \sim p(z)$  determines the high-level semantics, followed by (2) to produce the output sentences with low-level syntactic and lexical details. This contrasts with (1) in the explicit dependency on z.

Inference Similar to GPT-2, parameters  $\theta$  are typically learned by maximizing the marginal log likelihood  $\log p_{\theta}(x) = \log \int p(z) p_{\theta}(x|z) \mathrm{d}z$ . However, this marginal term is intractable to compute for many decoder choices. Thus, variational inference is considered, and the true posterior  $p_{\theta}(z|x) \propto p_{\theta}(x|z)p(z)$  is approximated via the variational distribution  $q_{\phi}(z|x)$  is (often known as the *inference model* or *encoder*), implemented via a  $\phi$ -parameterized neural network. It yields the *evidence lower bound objective* (ELBO):

<span id="page-2-1"></span>
$$\log p_{\theta}(x) \ge \mathcal{L}_{\text{ELBO}} =$$

$$\mathbb{E}_{q_{\phi}(z|x)} \left[ \log p_{\theta}(x|z) \right] - \text{KL}(q_{\phi}(z|x)||p(z))$$
(3)

Typically,  $q_{\phi}(z|x)$  is modeled as a Gaussian distribution, and the re-parametrization trick is used for efficient learning (Kingma and Welling, 2013).

A Taxonomy of Autoencoders There is an alternative interpretation of the ELBO: the VAE objective can be viewed as a regularized version of the autoencoder (AE) (Goodfellow et al., 2016). It is thus natural to extend the negative of  $\mathcal{L}_{ELBO}$  in (3) by introducing a hyper-parameter  $\beta$  to control the strength of regularization:

<span id="page-2-4"></span><span id="page-2-2"></span>
$$\mathcal{L}_{\beta} = \mathcal{L}_{E} + \beta \mathcal{L}_{R}, \text{ with}$$
 (4)

$$\mathcal{L}_{E} = -\mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} \left[ \log p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z}) \right]$$
 (5)

<span id="page-2-5"></span>
$$\mathcal{L}_R = \text{KL}(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p(\boldsymbol{z})) \tag{6}$$

where  $\mathcal{L}_E$  is the reconstruction error (or negative log-likelihood (NLL)), and  $\mathcal{L}_R$  is a KL regularizer. The cost function  $\mathcal{L}_\beta$  provides a unified perspective for understanding various autoencoder variants and training methods. We consider two types of latent space with the following objectives:

<span id="page-2-0"></span>• AE. Only  $\mathcal{L}_E$  is considered ( $\beta=0$ ), while the Gaussian sampling in  $q_{\phi}(z|x)$  remains. In other words, the regularization is removed, and a point-estimate is likely to be learned to represent the text sequence's latent feature. Note our reconstruction is on sentence-level, while other PLMs (Devlin et al., 2019; Yang et al., 2019) employ masked LM loss, performing token-level reconstruction.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 1: Illustration of OPTIMUS architecture.

• VAE. The full VAE objective is considered  $(\beta > 0)$ . It tends to learn a smooth latent space due to  $\mathcal{L}_R$ .

**Information Bottleneck Principle** From an information theory perspective, *information bottleneck* (IB) provides a principled approach to find the trade-off between *predictive power* and *complexity* (*compactness*) when summarizing observed data in learned representations. We show that our OPTIMUS pre-training objectives effectively practice the IB principle as follows.

The objective in (4) shows the  $\beta$ -VAE loss for one single sentence x. The training objective over the dataset q(x) can be written as:

$$\mathcal{F}_{\beta} = -\mathcal{F}_{E} + \beta \mathcal{F}_{R} \tag{7}$$

where  $\mathcal{F}_E = E_{q(\boldsymbol{x}), \boldsymbol{z} \sim q(\boldsymbol{z}|\boldsymbol{x})}[\log p(\tilde{\boldsymbol{x}}|\boldsymbol{z})]$  is the aggregated reconstruction term  $(\tilde{\boldsymbol{x}}$  is the reconstruction target), and  $\mathcal{F}_R = \mathbb{E}_{q(\boldsymbol{x})}[\mathrm{KL}(q(\boldsymbol{z}|\boldsymbol{x})||p(\boldsymbol{z}))]$  is the aggregated KL term. With the detailed proof shown in Section A of Appendix, we see that  $\mathcal{F}_\beta$  is an upper bound of IB:

$$\mathcal{F}_{\beta} \ge -I_{q}(\boldsymbol{z}, \tilde{\boldsymbol{x}}) + \beta I_{q}(\boldsymbol{z}, \boldsymbol{x}) = \mathcal{L}_{\text{IB}}, \quad (8)$$

where  $\mathcal{L}_{\mathrm{IB}}$  is the Lagrange relaxation form of IB presented by Tishby et al. (2000),  $I_q(\cdot,\cdot)$  is the mutual information (MI) measured by probability q. The goal of IB is to maximize the predictive power of z on target  $\tilde{x}$ , subject to the constraint on the amount of information about original x that z carries. When  $\beta=0$ , we have the AE variant of our OPTIMUS, the model fully focuses on maximizing the MI to recover sentences from the latent space. As  $\beta$  increases, the model gradually transits towards fitting the aggregated latent distribution  $q(z)=\int_x q(z|x)q(x)dx$  to the given prior p(z), leading the VAE variant of our OPTIMUS.

#### 4.2 Model Architectures

The model architecture of OPTIMUS is composed of multi-layer Transformer-based encoder and decoder, based on the original implementation described in (Vaswani et al., 2017). The overall architecture is illustrated in Figure 1. To leverage

the expressiveness power of existing PLMs, we initialize our encoder and decoder with weights of BERT  $\phi_{BERT}$  and GPT-2  $\theta_{GPT-2}$ , respectively. This procedure is seamless, as all of these models are trained in a self-supervised/unsupervised manner.

We denote the number of layers (*i.e.*, Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. Specifically, we consider BERT<sub>BASE</sub> (L=12, H=768, A=12, Total Parameters=110M) and GPT-2 (L=12, H=768, A=12, Total Parameters=117M). We hope that our approach can provide a practical recipe to inspire future work to integrate larger pre-trained encoder and decoder for higher performance models.

Connecting BERT & GPT-2 Two technical questions remain, when pre-training OPTIMUS from BERT & GPT-2: (i) How to represent sentences, since the two PLMs employ different tokenization schemes? (ii) How to adapt a pre-trained GPT-2 to arbitrary conditional input without retraining the model again? Controllable GPT-2 models have been studied in (Keskar et al., 2019; Zellers et al., 2019; Peng et al., 2020a,b) when prescribed control codes/tokens are provided, but it is still unknown how to ground GPT-2 to arbitrary conditional inputs.

**Tokenization** In BERT, WordPiece Embeddings (WPE) is used for tokenization (vocabulary size is 28996 for the cased version). In GPT-2, the modified Byte Pair Encoding (BPE) (Radford et al., 2019) is used for tokenization (vocabulary size is 50260). A given token is represented as  $h_{\rm Emb}$ , by summing the corresponding token, position and segment embeddings  $^3$ . For a sentence, we present it in both types of tokenization: the input of encoder is WPE, and the output of decoder is BPE to compute the reconstruction loss.

**Latent Vector Injection** Similar to BERT, the first token of every sentence is always a special classification token ([CLS]). The last-layer hidden state  $\boldsymbol{h}_{\text{[CLS]}} \in \mathbb{R}^H$  corresponding to this token is used as the sentence-level representation. It further constructs the latent representation  $\boldsymbol{z} = \mathbf{W}_{\text{E}}\boldsymbol{h}_{\text{[CLS]}}$ , where  $\boldsymbol{z} \in \mathbb{R}^P$  is a P-dimensional vector and  $\mathbf{W}_{\text{E}} \in \mathbb{R}^{P \times H}$  is the weight matrix. To facilitate  $\boldsymbol{z}$  in GPT-2 decoding without re-training the weights, we consider two schemes, illustrated in Figure 2:

<span id="page-3-1"></span><sup>&</sup>lt;sup>3</sup>OPTIMUS does not require segment embeddings, but we remain it due to BERT initialization.

<span id="page-4-1"></span>![](_page_4_Figure_0.jpeg)

Figure 2: Illustration of two schemes to inject latent vector. (a) Memory:  $x_t$  attends both  $x_{< t}$  and  $h_{\text{Mem}}$ ; (b) Embedding: latent embedding is added into old embeddings to construct new token embedding  $h'_{\text{Emb}}$ .

- Memory: z plays the role of an additional memory vector  $h_{\text{Mem}}$  for GPT2 to attend. Specifically,  $h_{\text{Mem}} = \mathbf{W}_{\mathbf{M}}z$ , where  $\mathbf{W}_{\mathbf{M}} \in \mathbb{R}^{LH \times P}$  is the weight matrix.  $h_{\text{Mem}} \in \mathbb{R}^{LH}$  is separated into L vectors of length H, each of which is attended by GPT-2 in one layer.
- Embedding: z is added on the original embedding layer, and directly used in every decoding step. The new embedding representation is  $h'_{\text{Emb}} = h_{\text{Emb}} + \mathbf{W}_{\text{D}}z$ , where  $\mathbf{W}_{\text{D}} \in \mathbb{R}^{H \times P}$ .

We study their empirical performance in Section B.1 of Appendix, and observe that Memory is significantly more effective than Embedding, and the integration of both schemes yields slightly better results. We hypothesize that the reason why Memory is superior is because it allows the decoder to attend the latent information at every layer of the network directly, while the Embedding method only allows the decoder to see the latent information at the input and output layer. In our experiments, we use the integration scheme by default. In summary, the encoder parameters  $\phi = \{\phi_{BERT}, W_E\}$ , and decoder parameters  $\theta = \{\theta_{GPT-2}, W_M, W_D\}$ .

### <span id="page-4-0"></span>4.3 Learning Procedures

We train the model parameters  $\{\phi, \theta\}$  using two objectives: AE and VAE, discussed in Section 4.1. Pre-training AE using (5) is straightforward. However, pre-training VAE can be challenging due to the notorious *KL vanishing* issue (Bowman et al., 2016), where (i) an encoder that produces posteriors almost identical to the Gaussian prior for all sentences (rather than a more interesting posterior); and (ii) a decoder that completely ignores z in (2), and a learned model that reduces to a simpler NLM.

To reduce this issue, we follow the intuition that if the encoder is providing useful information from the beginning of decoder training, the decoder is more likely to make use of z (Fu et al., 2019; He et al., 2019). Specifically, we use the cyclical schedule to anneal  $\beta$  for 10 periods (Fu et al., 2019). Within one period, there are three consecutive stages: Training AE ( $\beta=0$ ) for 0.5 proportion, annealing  $\beta$  from 0 to 1 for 0.25 proportion, and fixing  $\beta=1$  for 0.25 proportion. When  $\beta>0$ , we use the KL thresholding scheme (Li et al., 2019; Kingma et al., 2016), and replace the KL term  $\mathcal{L}_R$  in (6) with a hinge loss term that maxes each component of the original KL with a constant  $\lambda$ :

<span id="page-4-2"></span>
$$\mathcal{L}'_{R} = \sum_{i} \max[\lambda, \text{KL}(q_{\phi}(z_{i}|\boldsymbol{x})||p(z_{i}))]$$
 (9)

Here,  $z_i$  denotes the *i*th dimension of z. Using the thresholding objective causes learning to give up driving down KL for dimensions of z that are already beneath the target compression rate.

Pre-training data The pre-training procedure largely follows the existing literature on language model pre-training. We use English Wikipedia to pre-train our AE and VAE objectives. As our main interest is to model sentences (rather than text sequences of a fixed length), we pre-process Wikipedia with maximum sentences length 64. It leads to 1990K sentences, which accounts 96.45% Wikipedia sentences used in BERT. More data pre-processing details are in Section B.2 of Appendix.

### **5** Experimental Results

We consider to apply the pre-trained OPTIMUS models to three types of downstream tasks: (i) language modeling, where OPTIMUS is compared with SoTA VAE methods and GPT-2. (ii) Guided language generation, where OPTIMUS shows its unique advantage in producing controllable sentences in contrast to GPT-2. (iii) Low-resource language understanding, where the learned structured latent features can be used for fast adaptation in new tasks.

### **5.1** Language Modeling

Fine-tuning LM on new datasets is straightforward. We load the pre-trained OPTIMUS, and update the model with one additional  $\beta$  scheduling cycle for one epoch. The semantic latent vectors are first pre-trained off-the-shelf, and then easily leveraged to train the decoder on downstream datasets. From this perspective, our pre-training can be viewed as an effective approach to reduce KL vanishing.

<span id="page-5-0"></span>

|         | Dataset PTB      |        |      |      | YELP  |      |     | YAHOO |      |     | SNLI  |      |     |
|---------|------------------|--------|------|------|-------|------|-----|-------|------|-----|-------|------|-----|
|         |                  | LM     | Re   | pr.  | LM    | Re   | pr. | LM    | Re   | pr. | LM    | Re   | pr. |
|         | Method           | PPL↓   | MI ↑ | AU ↑ | PPL↓  | MI ↑ | AU↑ | PPL↓  | MI ↑ | AU↑ | PPL↓  | MI ↑ | AU↑ |
|         | $\lambda = 0.05$ | 23.58  | 3.78 | 32   | 21.99 | 2.54 | 32  | 22.34 | 5.34 | 32  | 13.47 | 3.49 | 32  |
| S []    | $\lambda = 0.10$ | 23.66  | 4.29 | 32   | 21.99 | 2.87 | 32  | 22.56 | 5.80 | 32  | 13.48 | 4.65 | 32  |
| ΙI      | $\lambda = 0.25$ | 24.34  | 5.98 | 32   | 22.20 | 5.31 | 32  | 22.63 | 7.42 | 32  | 14.08 | 7.22 | 32  |
| OPTIMUS | $\lambda = 0.50$ | 26.69  | 7.64 | 32   | 22.79 | 7.67 | 32  | 23.11 | 8.85 | 32  | 16.67 | 8.89 | 32  |
| _       | $\lambda = 1.00$ | 35.53  | 8.18 | 32   | 24.59 | 9.13 | 32  | 24.92 | 9.18 | 32  | 29.63 | 9.20 | 32  |
| ш       | M. A.            | 101.40 | 0.00 | 0    | 40.39 | 0.13 | 1   | 61.21 | 0.00 | 0   | 21.50 | 1.45 | 2   |
| VAE     | C. A.            | 108.81 | 1.27 | 5    |       |      |     | 66.93 | 2.77 | 4   | 23.67 | 3.60 | 5   |
|         | SA-VAE           |        |      |      |       | 1.70 | 8   | 60.40 | 2.70 | 10  |       |      |     |
| Small   | Aggressive       | 99.83  | 0.83 | 4    | 39.84 | 2.16 | 12  | 59.77 | 2.90 | 19  | 21.16 | 1.38 | 5   |
| 0,1     | AE-BP            | 96.86  | 5.31 | 32   | 47.97 | 7.89 | 32  | 59.28 | 8.08 | 32  | 21.64 | 7.71 | 32  |
|         | GPT-2            | 24.23  | -    | -    | 23.40 | -    | -   | 22.00 | -    | -   | 19.68 | -    | -   |
|         | LSTM-LM          | 100.47 | -    | -    | 42.60 | -    | -   | 60.75 | -    | -   | 21.44 | _    | -   |
|         | LSTM-AE          | -      | 8.22 | 32   | -     | 9.24 | 32  | -     | 9.26 | 32  | -     | 9.18 | 32  |

Table 1: Comparison on language modeling tasks on four datasets. "Small VAEs" indicate all previous language VAEs, which are built with two-layer LSTMs. All results for Small VAEs, LSTM-LM, LSTM-AE are quoted from literature, and GPT-2 results are produced by us. Best values are in blue.  $\lambda = 0.50$  is a good trade-off to achieve the best values on all metrics compared with small VAEs. "-" indicates the models are improper to report these values; Empty cells indicate the results were not reported in the literature.

We consider four datasets: the Penn Treebank (PTB) (Marcus et al., 1993), SNLI (Bowman et al., 2015), Yahoo, and Yelp corpora (Yang et al., 2017; He et al., 2019).

**Metrics** There are two types of metrics to evaluate language VAEs. (i) Generation capability: we use *perplexity* (PPL). Note that NLM and GPT-2 has exactly PPL, while VAEs does not. Following (He et al., 2019), we use the importance weighted bound in (Burda et al., 2015) to approximate  $\log p(x)$ , and report PPL. (ii) Representation learning capability: Active units (AU) of z and its Mutual Information (MI) with x. We report the full results with ELBO, KL and Reconstruction in Appendix, but note that higher ELBO does not necessarily yield better language modeling.

Baseline Methods (i) GPT-2. A large-scale LM trained on OpoenWebText (Radford et al., 2019). We load the pre-trained GPT-2 weights, and refine the model for 1 epoch on the new datasets. (ii) Annealing.  $\beta$  is gradually annealed from 0 to 1. This annealing procedure can be used once (M.A.) (Bowman et al., 2016) or multiple times (C.A.) (Fu et al., 2019). (iii) Aggressive Training (He et al., 2019). Training the encoder multiple times per decoder update. (iv) AE-FB (Li et al., 2019). Training AE, and then VAE using the KL thresholding in (9), the results on  $\lambda = 0.50$  are reported as a good trade-off.

The results are shown in Table 1. Various  $\lambda$ values are used, we observe a trade-off between language modeling and representation learning, controlled by  $\lambda$ . Compared with existing VAE methods, OPTIMUS achieve significantly lower perplexity, and higher MI/AU. This indicates that our pre-training method is an effective approach to reduce KL vanishing issue and training VAEs, especially given the fact that we only fine-tune on these datasets for one epoch. OPTIMUS achieves lower perplexity compared with GPT-2 on three out of four datasets. Intuitively, this is because the model can leverage the prior language knowledge encoded in z. This gap is larger, when the sentences in the dataset exhibit common regularities, such as SNLI, where the prior plays a more important/effective role in this scenario. Though the form of our model is simple, OPTIMUS shows stronger empirical performance than sophisticated models that are particularly designed for long-text, such as hVAE in (Shen et al., 2019). For example, the KL and PPL of OPTIMUS (15.09 and 22.79) are much better than hVAE (6.8 and 45.8) on Yelp dataset. This verifies the importance of pre-training a latent space. The full experimental results are shown in Table 8, 9, 10 and 11 of Appendix.

#### 5.2 Guided Language Generation

Different from the traditional NLMs or GPT-2, VAEs learns bidirectional mappings between the

<span id="page-6-0"></span>

| Source $x_A$ a girl makes a silly face                                                                                                                                                                                                                                             | Target $x_B$<br>two soccer players are playing soccer                                                                                                                                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <ul> <li>Input x<sub>C</sub></li> <li>a girl poses for a picture</li> <li>a girl in a blue shirt is taking pictures of a microscope</li> <li>a woman with a red scarf looks at the stars</li> <li>a boy is taking a bath</li> <li>a little boy is eating a bowl of soup</li> </ul> | <ul> <li>Output x<sub>D</sub></li> <li>two soccer players are at a soccer game.</li> <li>two football players in blue uniforms are at a field hockey game</li> <li>two men in white uniforms are field hockey players</li> <li>two baseball players are at the baseball diamond</li> <li>two men are in baseball practice</li> </ul> |

Table 2: Sentence transfer via arithmetic  $z_D = z_B - z_A + z_C$ . The output sentences are in blue.

<span id="page-6-1"></span>

| I | 0.0 | children are looking for the water to be clear.      |
|---|-----|------------------------------------------------------|
| I | 0.1 | children are looking for the water.                  |
| I | 0.2 | children are looking at the water.                   |
| I | 0.3 | the children are looking at a large group of people. |
| I | 0.4 | the children are watching a group of people.         |
| I | 0.5 | the people are watching a group of ducks.            |
| I | 0.6 | the people are playing soccer in the field.          |
| I | 0.7 | there are people playing a sport.                    |
| I | 0.8 | there are people playing a soccer game.              |
| I | 0.9 | there are two people playing soccer.                 |
| I | 1.0 | there are two people playing soccer.                 |
| ı |     |                                                      |

Table 3: Interpolating latent space  $z_{\tau} = z_1 \cdot (1 - \tau) + z_2 \cdot \tau$ . Each row shows  $\tau$ , and the generated sentence (in blue) conditioned on  $z_{\tau}$ .

latent and symbolic space. It enables high-level sentence editing as arithmetic latent vector operations, and thus allows guided language generation. The reason that Optimus supports arithmetic operations are two-fold: (1) Pre-training on large datasets with large networks allows all sentences to be densely and faithfully represented in the latent space. (2) The continuity property of neural nets and KL regularization of VAE encourage latent vectors with similar semantics are smoothly organized together.

This is demonstrated with two simple schemes to manipulate pre-trained latent spaces: sentence transfer and interpolation, with results in Table 2 and Table 3, respectively. Details and more results are shown in Appendix. They showcase that OPTIMUS enables new ways that one can play with language generation using pre-trained models, compared with GPT-2 that can only fulfill text sequences with given prompts. A website demo<sup>4</sup> is released to the public to interact with the model, exhibiting the power of latent-vector-based controllable text generation. We demonstrate more sophisticated ways to manipulate pre-trained latent spaces in three real applications as follows.

<span id="page-6-3"></span>

| Metrics    | Seq2Seq | CVAE  | WAE   | iVAE <sub>MI</sub> | OPTIMUS |
|------------|---------|-------|-------|--------------------|---------|
| Recall↑    | 0.232   | 0.265 | 0.289 | 0.355              | 0.362   |
| Precision↑ | 0.232   | 0.222 | 0.266 | 0.239              | 0.313   |
| F1↑        | 0.232   | 0.242 | 0.277 | 0.285              | 0.336   |

Table 4: Dialog response generation on DailyDialog dataset. All numbers are from (Gu et al., 2019) except that iVAE<sub>MI</sub> is from (Fang et al., 2019).

<span id="page-6-4"></span>

| Methods     | Recall↑ | Precision↑ | F1↑   | Neural↑ | N-gram↑ |
|-------------|---------|------------|-------|---------|---------|
| StyleFusion | 0.374   | 0.242      | 0.294 | 0.1050  | 0.1495  |
| OPTIMUS     | 0.385   | 0.268      | 0.316 | 0.1191  | 0.1645  |

Table 5: Stylized response generation.

<span id="page-6-5"></span>

| Metrics    | Control-Gen | ARAE  | NN-Outlines | OPTIMUS |
|------------|-------------|-------|-------------|---------|
| Accuracy ↑ | 0.878       | 0.967 | 0.553       | 0.998   |
| Bleu ↑     | 0.389       | 0.201 | 0.198       | 0.398   |
| G-score↑   | 0.584       | 0.442 | 0.331       | 0.630   |
| Self-Bleu↓ | 0.412       | 0.258 | 0.347       | 0.243   |

Table 6: Label-conditional text generation on Yelp.

**Dialog response generation** The open-domain dialog response generation task is considered: generating responses x given a dialog history c. Following (Gao et al., 2019a), we embed the history and response in a joint latent space as  $z_{S2S}$  and  $z_{\rm AE}$ , respectively. A fusion regularization is used to match the responses to the context. We consider Dailydialog (Li et al., 2017c) used in (Gu et al., 2019), which has 13,118 daily conversations. Each utterance is processed as the response of previous 10 context utterances from both speakers. The baseline methods are described in Appendix. We measure the performance using Bleu (Chen and Cherry, 2014), and compute the precision, recall and F1 in Table 4. OPTIMUS shows higher Bleu scores than all existing baselines.

**Stylized response generation** Following Style-Fusion (Gao et al., 2019b), we consider generating responses for Dailydialog in the style of Holmes. The comparison is shown in Table 5. In addition to Bleu, we use neural and N-gram classifier scores

<span id="page-6-2"></span><sup>4</sup>http://aka.ms/optimus

<span id="page-7-0"></span>

| System<br>Dataset size |                                 | MNLI<br>392k   | QQP<br>363k             | QNLI<br>108k   | SST-2<br>67k   | CoLA<br>8.5k   | STS-B<br>5.7k           | MRPC<br>3.5k   | RTE<br>2.5k    | WNLI<br>634             | Average                                                           |
|------------------------|---------------------------------|----------------|-------------------------|----------------|----------------|----------------|-------------------------|----------------|----------------|-------------------------|-------------------------------------------------------------------|
| Feature-based          | BERT OPTIMUS (VAE) OPTIMUS (AE) | 0.468          | 0.146<br>0.662<br>0.565 | 0.720          |                |                | 0.690<br>0.719<br>0.655 |                |                | 0.577<br>0.563<br>0.620 | $0.531 \pm 0.011$<br>$0.607 \pm 0.013$<br>$0.569 \pm 0.010$       |
| Fine-tuning            | BERT<br>OPTIMUS (VAE)           | 0.835<br>0.834 | 0.909<br>0.909          | 0.912<br>0.908 | 0.923<br>0.924 | 0.598<br>0.573 | 0.886<br>0.888          | 0.868<br>0.873 | 0.700<br>0.697 | 0.507<br>0.563          | $\begin{array}{c} 0.793 \pm 0.008 \\ 0.798 \pm 0.017 \end{array}$ |

Table 7: Comparison of BERT and OPTIMUS (with the AE and VAE objectives). Comparison is on the validation set of GLUE. F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks.

to evaluate the accuracy of the generated responses that belong to the desired style. OPTIMUS achieves better performance on all metrics.

Label-conditional text generation The short Yelp dataset collected in (Shen et al., 2017) is used. It contains 444K training sentences, and we use separated datasets of 10K sentences for validation/testing, respectively. The goal is to generate text reviews given the positive/negative sentiment. We fine-tune OPTIMUS using the VAE objective on the dataset, then freeze backbone weights. A conditional GAN (Mirza and Osindero, 2014) is trained on the fixed latent space. The generation process is to first produce a latent vector  $z_y$  based on a given label y using conditional GAN, then generate sentences conditioned on  $z_u$  using the decoder. The baselines are described in Appendix. G-score computes the geometric mean of Accuracy and Bleu, measuring the comprehensive quality of both content and style. Self-Bleu measures the diversity of the generated sentences. The results are shown in Table 6, OPTIMUS achieves the best performance on all metrics. This verifies the importance of learning a smooth and meaningful latent space. The conditional generated sentences are shown in Appendix.

#### 5.3 Low-resource Language Understanding

Due to the regularization term  $\mathcal{L}_R$ , OPTIMUS can organize sentences in the way specified by the prior distribution. For basic VAEs, a smooth feature space is learned, which is specifically beneficial for better generalization when the number of task-specific labeled data is low. To have a fair comparison, we follow the BERT paper, where the hidden feature of [CLS] is used as the sentence-level representation. In this way, the linear classifiers for both models have the same number of trainable parameters. Though the latent vector z is

typically used as sentence-level representation in VAE literature, we argue that the KL regularization applied on z has a large impact on the preceding layer feature  $\boldsymbol{h}_{\text{[CLS]}}$ . Specifically,  $\boldsymbol{h}_{\text{[CLS]}}$  is fed into an linear classifier  $\mathbf{W}_{\text{C}} \in \mathbb{R}^{K \times H}$ , where K is the number of classes, with objective  $-\log(\operatorname{softmax}(\boldsymbol{h}_{\text{[CLS]}}\mathbf{W}_{\text{C}}^{\top}))$ . Two schemes are used: (i) Fine-tuning, where both the pre-trained model and the classifier are updated; (ii) Feature-based, where pre-trained model weights are frozen to provide embeddings for the classifier update.

Sentiment classification on Yelp dataset. A varying number of training samples are randomly chosen, ranging from 1 to 10K per class. 10 trials are used when the number of available training samples are small, each is trained in 100 training epochs. The results are shown in Figure 3. When pre-trained models are used to provide sentence embeddings, the proposed OPTIMUS consistently outperforms BERT. It demonstrates that the latent structure learned by OPTIMUS is more separated, and helps generalize better. When the entire network is fine-tuned, OPTIMUS can adapt faster than BERT, when the available number of training samples is small. The two methods perform quite similarly when more training data is provided. This is because the pre-trained backbone network size is much larger than the classifier, where the performance is dominated by the backbone networks.

**Visualization of the latent space.** We use tSNE (Maaten and Hinton, 2008) to visualize the learned feature on a 2D map. The validation set of Yelp is used to extract the latent features. Compared with BERT, OPTIMUS learns a smoother space and more structured latent patterns, which explains why OPTIMUS can yield better classification performance and faster adaptation.

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 3: Testing accuracy with a varying number of labeled training samples per class on the Yelp dataset.

![](_page_8_Figure_2.jpeg)

Figure 4: Comparison of tSNE visualization for the learned features. The colors indicate different labels.

**GLUE.** We further consider the GLUE benchmark (Wang et al., 2019), which consists of nine datasets for general language understanding. Following the finetuning schedule in (Devlin et al., 2019), we use learning rate  $[2,3,4,5] \times 10^{-5}$  and train the model for 3 epochs. We select the best performance among different runs. We show the results on the validation set in Table 7. With the feature-based scheme, OPTIMUS yields higher performance than BERT, especially on the large datasets such as MNLI, QQP and QNLI. When the full models are fine-tuned, the two methods perform quite similarly.

In summary, the scenarios that OPTIMUS fit the low-resource settings are two-fold: (1) The required computing resource is low: the feature-based approach only updates the classifier, whose computing requirement is much lower than full-model fine-tuning; (2) The number of required labelled data is low: when labelled data is rare, OPTIMUS adapts better. The results confirm that OPTIMUS can maintain and exploit the structures learned in pre-training, and presents a more general representation that can be adapted to new tasks more easily than BERT – feature-based adaption is much faster and easier to perform than fine-tuning.

### 6 Discussion

We present OPTIMUS, a large-scale pre-trained deep latent variable model for natural language. It introduces a smooth and universal latent space, by combining the advantages of VAEs, BERT and GPT-2 in one model. Experimental results on a wide range of tasks and datasets have demonstrated the strong performance of OPTIMUS, including new state-of-the-art for language VAEs.

There are several limitations in current OPTI-MUS. First, our pre-trained language VAE is still under-trained due to limited compute resource, as the training reconstruction loss can still decrease. One may further train the models with higher latent dimension and longer time to fully release the power of pre-trained latent spaces. Second, the current model can only control sentences of moderate length. One future direction is to consider more sophisticated mechanisms to gain stronger controlability over longer sentences while maintaining the compactness of latent representations.

While deep generative models (DGMs) such as VAEs are theoretically attractive due to its principle nature, it is now rarely used by practitioners in the modern pre-trained language modeling era where BERT/GPT dominate with strong empirical performance. That's why this paper makes a timely contribution to making DGMs practical for NLP. We hope that this paper will help renew interest in DGMs for this purpose. Hence, we deliberately keep a simple model, believing that the first pretrained big VAE model itself and its implications are novel: it helps the community to recognize the importance of DGMs in the pre-training era, and revisit DGMs to make it more practical. Indeed, OPTIMUS is uniquely positioned to learn a smooth latent space to organize sentences, which can enable guided language generation compared with GPT-2, and yield better generalization in lowresource language understanding tasks than BERT.
