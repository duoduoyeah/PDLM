## <span id="page-12-3"></span>**B** Pre-training Details

### <span id="page-12-1"></span>**B.1** Latent Vector Injection Schemes

We compare three different schemes to inject latent vector into GPT2 in Figure 5:

- **Mem**. Latent vector z is used as additional *memory* token for GPT2 to attend.
- **Emb**. Latent vector z is used as additional *embedding* to add into other embeddings.
- **Mem+Emb**. The integration of the above two schemes.

On both Yelp and PTB datasets, 5 training epochs are considered. Yelp generally has longer sentences than PTB. The encoder is initialized with BERT, and decoder is initialized with GPT-2. Lower reconstruction error per word indicates a more effective approach to pass the information flow from encoder to decoder. We see that it is significantly more efficient to use z as a memory vector for GPT-2 to attend, than as the additional embedding. The combined scheme yields slightly better performance in the late stage of training. In the paper, we use the combined scheme in default.

### <span id="page-12-2"></span>**B.2** Wikipedia Dataset

We illustrate the statistics of Wikipedia dataset in Figure 6. Since we focus on modeling natural sentences (rather than text sequences of a fixed length as in GPT-2 (Radford et al., 2019)) in a latent space, we pre-process Wikipedia into a set of natural sentences, with maximum sequence length as 64. This leads to 1990K sentences, which is 96.45% of entire Wikipedia dataset.

### **C** Experiment Details

#### C.1 Language Modeling

In addition to generating high-quality sentences as in the traditional language models that only, VAEs also aim to learn a good posterior distribution in the latent space. The language modeling performance

<span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

Figure 5: Illustration of three different schemes to inject latent vector into GPT-2 for guided language generation: (a) Yelp and (b) PTB. The learning curves for reconstruction error per word is considered. Emb indicates latent vector is used as additional embedding to add into other embeddings, and Mem indicates latent vector is used as additional memory token for GPT2 to attend. Mem+Emb indicates the integration of two schemes.

<span id="page-13-1"></span>![](_page_13_Figure_2.jpeg)

Figure 6: Illustration of sentence distribution in Wikipedia dataset: (a) Frequency distribution and (b) Cumulative Frequency distribution. We choose maximum length as 64 to construct the pre-training dataset. It leads to 1990K sentences, which is 96.45% of entire Wikipedia dataset.

is evaluated with ELBO, perplexity (PPL) or importance weighted perplexity (He et al., 2019), which provides a tighter bound to  $\log p(x)$ . Higher ELBO and lower PPL indicate the model fits the observed sentences better. The pre-training takes around 50 hours for one epoch on eight V100 DGX2 GPU's.

- **ELBO**: The sum of KL divergence and reconstruction loss.
- **Perplexity.** PPL =  $p(x_1, \dots, x_N)^{-1/N}$ , where N is the number of words. For latent variable models, we use a lower bound on the marginal log-likelihood  $\log p(x)$ , as follows from Jensen's Inequality and the fact that the average importance weights are an unbiased estimator of p(x):

$$\mathcal{L}_{k} = \mathbb{E}\left[\log \frac{1}{k} \sum_{i=1}^{k} w_{i}\right]$$

$$\leq \log\left[\mathbb{E}\frac{1}{k} \sum_{i=1}^{k} w_{i}\right] = \log p(\boldsymbol{x}). \tag{15}$$

where 
$$w_i = p(\boldsymbol{x}, \boldsymbol{z}_i)/q(\boldsymbol{z}_i|\boldsymbol{x})$$
.

More importantly, we are interested in the learned z, which is evaluated using the following three metrics:

- AU: The total number of active units in z, defined as  $A_z = \text{Cov}_{\boldsymbol{x}}(\mathbb{E}_{z \sim q(z|\boldsymbol{x})}[z]) > 0.01$  (Burda et al., 2015);
- MI: The mutual information I(x, z);
- KL: The posterior-prior KL divergence

The full experimental results on shown in Table 8, 9, 10 and 11.

### C.2 Dialog response generation

**Dialog response generation: SpaceFusion** We interpolate samples  $z_{\tau}$  between the context and response as  $z_{\tau} = \tau z_{\text{S2S}} + (1 - \tau) z_{\text{AE}}$ , where  $\tau \sim$  Uniform(0, 1). We fix the first 11 layers of encoder, and fine-tune from last layer to z:  $\{\phi_{\text{AE}}, \phi_{\text{E}}\}$ . An additional network path  $\{\phi_{\text{S2S}}, \phi_{\text{E}}'\}$  is introduced

<span id="page-14-0"></span>

| Metric                   | LM     | Repre | sentation | Learni | ng Objec | ctive  |
|--------------------------|--------|-------|-----------|--------|----------|--------|
| Method                   | PPL ↓  | MI↑   | AU↑       | -ELBO↓ | KL↑      | Rec↓   |
| Ours( $\lambda = 0.05$ ) | 23.58  | 3.78  | 32        | 91.31  | 4.88     | 86.43  |
| $Ours(\lambda = 0.1)$    | 23.66  | 4.29  | 32        | 91.60  | 5.82     | 85.78  |
| $Ours(\lambda = 0.25)$   | 24.24  | 5.98  | 32        | 93.18  | 9.42     | 83.75  |
| $Ours(\lambda = 0.5)$    | 26.69  | 7.64  | 32        | 96.82  | 15.72    | 81.09  |
| $Ours(\lambda = 1.0)$    | 35.53  | 8.18  | 32        | 77.65  | 28.50    | 77.65  |
| GPT-2                    | 24.23  |       |           |        |          |        |
| LSTM-LM                  | 100.47 |       |           | 101.04 |          |        |
| LSTM-AE                  |        | 8.22  | 32        |        |          | 70.36  |
| M. Annealing             | 101.40 | 0.0   | 0         | 101.28 | 0.0      | 101.28 |
| C. Annealing             | 108.81 | 1.27  | 5         | 102.81 | 1.37     | 101.85 |
| Aggressive               | 99.83  | 0.83  | 4         | 101.19 | 0.93     | 100.26 |
| AE-BP ( $\lambda = 5$ )  | 96.86  | 5.31  | 32        | 102.41 | 6.54     | 95.87  |

Table 8: Comparison on PTB dataset.

<span id="page-14-1"></span>

| Metric                   | LM    | Repre | sentation | Learni | ng Obje | ctive  |
|--------------------------|-------|-------|-----------|--------|---------|--------|
| Method                   | PPL ↓ | MI ↑  | AU↑       | -ELBO↓ | KL↑     | Rec ↓  |
| Ours( $\lambda = 0.01$ ) | 21.99 | 2.54  | 32        | 337.41 | 3.09    | 334.31 |
| $Ours(\lambda = 0.05)$   | 21.99 | 2.87  | 32        | 337.61 | 3.73    | 333.87 |
| $Ours(\lambda = 0.25)$   | 22.20 | 5.31  | 32        | 340.03 | 8.70    | 331.33 |
| $Ours(\lambda = 0.5)$    | 22.79 | 7.67  | 32        | 344.10 | 15.09   | 329.01 |
| $Ours(\lambda = 1.0)$    | 24.59 | 9.13  | 32        | 353.67 | 27.89   | 325.77 |
| GPT-2                    | 23.40 |       |           |        |         |        |
| LSTM-LM                  |       |       |           | 358.10 |         |        |
| LSTM-AE                  |       | 9.26  | 32        |        |         | 278.76 |
| SA-VAE                   |       | 1.7   | 8         | 355.90 | 2.80    | 353.10 |
| M. Annealing             | 40.39 | 0.13  | 1         | 357.76 | 0.14    | 357.62 |
| C. Annealing             |       |       |           |        |         |        |
| Aggressive               |       | 2.4   | 7         | 328.40 | 3.4     | 322.70 |
| AE-BP ( $\lambda = 5$ )  |       |       |           |        |         |        |

Table 9: Comparison on Yelp dataset. For LSTM-LM and GPT-2, we report the exact negative log likelihood.

from the 11th layer of encoder to z to represent context. The fine-tuning objective is:

$$\min_{\{\phi_{\text{S2S}},\phi_{\text{AE}},\phi_{\text{E}},\phi_{\text{E}}',\theta\}} \mathcal{L}_{\text{dialog}} = \mathcal{L}_{\bm{x}} + \mathcal{L}_{\text{fusion}}$$

where  $\mathcal{L}_{\text{fusion}}$  is the same with fusion term in (Gao et al., 2019a), and  $\mathcal{L}_{\boldsymbol{x}} = -[\log p(\boldsymbol{x}|\boldsymbol{z}_{\text{S2S}}) + \log p(\boldsymbol{x}|\boldsymbol{z}_{\text{AE}}) + \log p(\boldsymbol{x}|\boldsymbol{z}_{\text{T}})].$ 

We benchmark representative baselines and state-of-the-art approaches, including: (i) Seq2Seq: a generalized sequence-to-sequence model with hierarchical RNN encoder (Serban et al., 2016); (ii) SeqGAN: a GAN based model for sequence generation (Li et al., 2017b); (iii) CVAE baseline (Zhao et al., 2017); (iv) Dialogue WAE, a condi-

tional Wasserstein auto-encoder for response generation (Gu et al., 2019); ( $\nu$ ): A hierarchical VAE model (Serban et al., 2017). ( $\nu$ i) VHCR: a hierarchical VAE model with conversation modeling (Park et al., 2018). ( $\nu$ ii) iVAE<sub>MI</sub>: An implicit VAE model augmented with mutual information regularizer (Fang et al., 2019). The full comparison in shown in Table 12.

Stylized response generation: StyleFusion In this task, the additional sentences b are used to bias the generated response towards the reference style. The biased response representation is  $z_{\tau}' = \tau z_{\text{Style}} + (1 - \tau) z_{\text{AE}}$ , where  $\tau \sim \text{Uniform}(0, 1)$  and  $z_{\text{Style}}$  is the latent representation of b. The

<span id="page-15-0"></span>

| Metric                   | LM    | Repre | sentation | Learni | ng Obje | ctive  |
|--------------------------|-------|-------|-----------|--------|---------|--------|
| Method                   | PPL ↓ | MI ↑  | AU↑       | -ELBO↓ | KL↑     | Rec ↓  |
| Ours( $\lambda = 0.05$ ) | 22.34 | 5.34  | 32        | 282.70 | 6.97    | 282.84 |
| $Ours(\lambda = 0.10)$   | 22.56 | 5.80  | 32        | 289.88 | 7.77    | 282.11 |
| $Ours(\lambda = 0.25)$   | 22.63 | 7.42  | 32        | 290.69 | 11.19   | 279.49 |
| $Ours(\lambda = 0.50)$   | 23.11 | 8.85  | 32        | 293.34 | 17.45   | 275.89 |
| $Ours(\lambda = 1.0)$    | 24.92 | 9.18  | 32        | 301.21 | 30.41   | 270.80 |
| GPT-2                    | 22.00 |       |           |        |         |        |
| LSTM-LM                  | 60.75 |       |           | 328.00 |         |        |
| LSTM-AE                  |       | 9.26  | 32        |        |         | 278.76 |
| SA-VAE                   | 60.40 | 2.70  | 10        | 327.20 | 5.20    | 325.00 |
| M. Annealing             | 61.21 | 0.0   | 0         | 328.80 | 0.0     | 328.80 |
| C. Annealing             | 64.26 | 0.0   | 1         | 332.68 | 0.03    | 332.65 |
| Aggressive               | 59.77 | 2.9   | 15        | 328.40 | 5.70    | 322.70 |
| AE-BP ( $\lambda = 5$ )  | 59.28 | 8.08  | 32        | 329.31 | 10.76   | 318.55 |

Table 10: Comparison on Yahoo dataset.

<span id="page-15-1"></span>

| Metric                                    | LM    | Repre | sentation | Learnir | ng Objec | tive  |
|-------------------------------------------|-------|-------|-----------|---------|----------|-------|
| Method                                    | PPL ↓ | MI ↑  | AU↑       | -ELBO↓  | KL↑      | Rec ↓ |
| Ours( $\lambda = 0.05$ )                  | 13.47 | 3.49  | 32        | 33.08   | 3.92     | 29.17 |
| $Ours(\lambda = 0.10)$                    | 13.48 | 4.65  | 32        | 33.45   | 5.44     | 28.01 |
| Ours( $\lambda = 0.25$ )                  | 14.08 | 7.22  | 32        | 35.04   | 9.79     | 25.25 |
| $Ours(\lambda = 0.50)$                    | 16.67 | 8.89  | 32        | 38.50   | 16.35    | 22.14 |
| $Ours(\lambda = 1.00)$                    | 29.63 | 9.20  | 32        | 47.35   | 28.96    | 18.39 |
| GPT-2 (Radford et al., 2019)              | 20.24 |       |           |         |          |       |
| LSTM-LM                                   | 21.44 |       |           |         |          |       |
| LSTM-AE                                   |       | 9.18  | 32        |         |          |       |
| M. Annealing (Bowman et al., 2016)        | 21.50 | 1.42  | 2         | 33.07   | 1.42     | 31.66 |
| C. Annealing (Fu et al., 2019)            | 21.62 | 2.33  | 4         | 33.25   | 2.36     | 30.89 |
| Aggressive (He et al., 2019)              | 21.16 | 1.38  | 5         | 32.95   | 1.42     | 31.53 |
| AE-BP ( $\lambda = 5$ ) (Li et al., 2019) | 21.64 | 7.71  | 32        | 34.47   | 9.53     | 24.94 |

Table 11: Comparison on SNLI dataset. For LSTM-LM and GPT-2, we report the exact negative log likelihood.

corresponding loss for the biased target is  $\mathcal{L}_{x}' = -[\tau \log p(x|\mathbf{z}_{\text{Style}}) + (1-\tau) \log p(x|\mathbf{z}_{\text{AE}})]$ , which is added into  $\mathcal{L}_{\text{dialog}}$  for training.

**Evaluation** Two type of *Accuracy* are reported, based on text sequence (*i.e.*, neural) and its N-gram information. The accuracy is assessed by an oracle classifier to correctly predict whether generated response belongs the style-reference dataset.
