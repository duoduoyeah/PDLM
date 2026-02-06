## **C.2.1** Label-Conditional Text Generation

The goal of this task is to generate sentences conditioned on a given label. We consider a two-stage algorithm to adapt OPTIMUS for this task. First,

we fine-tune a VAE language model on the down-stream dataset, and freeze the model parameters. In another word, the latent space is fixed. Second, we build a conditional GAN for the latent space. Let's denote the latent vectors for ground-trurh sentences as  $\boldsymbol{z}_{\text{true}}$ . We build a generator G to produce  $\boldsymbol{z}_{\text{fake}} = G(\epsilon, y)$ , where  $\epsilon$  is the random noise, and y is the label. A discriminator D is trained simultaneously to distinguish  $\boldsymbol{z}_{\text{true}}$  and  $\boldsymbol{z}_{\text{fake}}$ . The learning objectives for conditional GAN is:

$$\min_{G} \max_{D} \mathcal{L}_{cGAN}$$

$$= \mathbb{E}_{\boldsymbol{x}, y \sim q(\boldsymbol{x}, y)} \left[ \mathbb{E}_{\boldsymbol{z} \sim q(\boldsymbol{z} | \boldsymbol{x})} [\log p_D(d = 1 | E(\boldsymbol{x}))] \right]$$

$$+ \mathbb{E}_{\epsilon \sim p_0(\epsilon)} [\log p_D(d = 0 | G(\epsilon, y))] \right] \tag{16}$$

To make the model work effectively, it is key to learn a smooth and meaningful latent space of target sentences. The text generation procedure conditioned on label y is:

$$x \sim p_{\theta}(x|z), \text{ with } z = G(\epsilon, y)$$
 (17)

This mimics the process to produce the outlines of the sentences using conditional GAN, and fill in details using the decoder. We show some generated sentences in Table [20.](#page-21-0)

We compare with three baselines: (1) *Ctrl-Gen* [\(Hu et al.,](#page-9-5) [2017\)](#page-9-5); We use their released code to reproduce the results. (2) *ARAE* [\(Zhao et al.,](#page-11-9) [2018\)](#page-11-9) proposes to learn an auto-encoder first, and then train a GAN to produce the latent vectors. (3) *NN-Outlines* [\(Subramanian et al.,](#page-10-10) [2018\)](#page-10-10) proposes the use of a general purpose encoder for text generation, and we implement it using BERT. Note that our two-stage fine-tuning scheme borrows the ideas from ARAE and NN-Outlines. The key difference is that we employ our pre-trained OPTIMUS model, and work on a better latent space.

Evaluation We consider three metrics: (1) *Bleu* for sentence quality, (2) *Accuracy* for conditional generation capability. The accuracy is assessed by an oracle classifier to correctly predict the attributes that generated sentences are conditioned on. (3) *G-score* is reported as the geometric mean of Accuracy and Bleu. This is the most important metric, as it evaluates the overall performance. For label-conditional text generation, Bleu of each generated sentence is computed by comparing with all sentences in the test set, as there are no source sentences. We further report Self-Bleu [\(Zhu et al.,](#page-11-10) [2018\)](#page-11-10) to evaluate the diversity of generated sentences.
