---
title: 'Transformer and ViT dataflow notation'
date: 2021-01-15
permalink: /posts/2021/01/vit-transformation-notation/
tags:
  - Vision Transformer
  - ViT
  - Deep Learning
  - Computer Vision
  - Machine Learning
  - Transformers
  - Mathematical Notation
  - nanoGPT
published: True 
mathjax: true    
---



When reviewing the transformer and ViT literature, to get an intuitive understanding of the various model layers and how the input tokens are manipulated, I found it helpful to map out the data flow through the model in matrix notation. I couldn't find this anywhere so I thought I'd share it.

---

# Part I: NanoGPT Foundation

This section covers the matrix notation for nanoGPT.

## Notation / sizes (nanoGPT)

- Batch $$B$$, sequence length $$T$$, vocab size $$V$$
- Model width $$D$$ (aka $$d_{\text{model}}$$)
- MLP hidden width $$d_{\text{ff}}$$ (often $$4D$$)
- Single head so $$d_k=d_v=D$$ (keeps shapes simple)
- Pre-LayerNorm residual layout (GPT-2/NanoGPT style)

---

## 0) Inputs, embeddings, positions

- Token indices: $$\mathbf{t}\in\{1,\dots,V\}^{B\times T}$$
- Token embedding matrix: $$E\in\mathbb{R}^{V\times D}$$
- Learned positional embeddings: $$P\in\mathbb{R}^{T\times D}$$

**Lookup + add positions**
$$X^{(0)} \;=\; E[\mathbf{t}] + P \quad\in\; \mathbb{R}^{B\times T\times D}$$

---

## 1) A single Transformer block $$\ell=1,\dots,L$$

### 1.1 Pre-norm (attention)

LayerNorm acts per token (last dim):
$$\tilde{X} \;=\; \mathrm{LN}^{(\ell)}_{\mathrm{attn}}\!\left(X^{(\ell-1)}\right) \;\in\; \mathbb{R}^{B\times T\times D}$$

(Each token vector $$x\in\mathbb{R}^{D}$$ is normalized and then scaled/shifted by $$\gamma,\beta\in\mathbb{R}^{D}$$.)

### 1.2 Single-head Q/K/V projections

Weights $$W_Q^{(\ell)},W_K^{(\ell)},W_V^{(\ell)}\in\mathbb{R}^{D\times D}$$:
$$Q \;=\; \tilde{X}W_Q^{(\ell)}, \qquad K \;=\; \tilde{X}W_K^{(\ell)}, \qquad V \;=\; \tilde{X}W_V^{(\ell)} \quad\in\; \mathbb{R}^{B\times T\times D}$$

### 1.3 Scaled dot-product attention (causal)

Scores (pairwise dot products per batch):
$$S \;=\; \frac{QK^{\top}}{\sqrt{D}} \;+\; M \quad\in\; \mathbb{R}^{B\times T\times T}$$

where $$M_{ij}=0$$ if $$j\le i$$ and $$M_{ij}=-\infty$$ if $$j>i$$ (upper-triangular causal mask).

Row-wise softmax:
$$A \;=\; \mathrm{softmax}(S) \;\in\; \mathbb{R}^{B\times T\times T}$$

Weighted sum of values:
$$H \;=\; A\,V \;\in\; \mathbb{R}^{B\times T\times D}$$

Output projection $$W_O^{(\ell)}\in\mathbb{R}^{D\times D}$$:
$$O \;=\; H\,W_O^{(\ell)} \;\in\; \mathbb{R}^{B\times T\times D}$$

Residual add:
$$X' \;=\; X^{(\ell-1)} + O \;\in\; \mathbb{R}^{B\times T\times D}$$

### 1.4 Pre-norm (MLP)

$$\hat{X} \;=\; \mathrm{LN}^{(\ell)}_{\mathrm{mlp}}(X') \;\in\; \mathbb{R}^{B\times T\times D}$$

### 1.5 MLP with GELU

Weights $$W_1^{(\ell)}\in\mathbb{R}^{D\times d_{\text{ff}}}$$, $$W_2^{(\ell)}\in\mathbb{R}^{d_{\text{ff}}\times D}$$ (biases optional):
$$U \;=\; \hat{X}W_1^{(\ell)} + b_1^{(\ell)} \;\in\; \mathbb{R}^{B\times T\times d_{\text{ff}}}$$

$$G \;=\; \mathrm{GELU}(U) \;\in\; \mathbb{R}^{B\times T\times d_{\text{ff}}}$$

$$M \;=\; G\,W_2^{(\ell)} + b_2^{(\ell)} \;\in\; \mathbb{R}^{B\times T\times D}$$

Residual add:
$$X^{(\ell)} \;=\; X' + M \;\in\; \mathbb{R}^{B\times T\times D}$$

(Repeat this block for $$\ell=1,\dots,L$$.)

---

## 2) Output head $$\rightarrow$$ logits

Final LayerNorm:
$$X_f \;=\; \mathrm{LN}_f\!\left(X^{(L)}\right) \;\in\; \mathbb{R}^{B\times T\times D}$$

Linear head to vocab. Two common options:

**(a) Weight tying (GPT-style):**
$$Z \;=\; X_f\,E^{\top} \;\in\; \mathbb{R}^{B\times T\times V}$$

**(b) Separate head:**
$$Z \;=\; X_f\,W_{\text{vocab}} + b_{\text{vocab}}, \qquad W_{\text{vocab}}\in\mathbb{R}^{D\times V}$$

These logits $$Z$$ give the next-token distribution via softmax over the vocab dimension; training uses cross-entropy with targets shifted by one.

---

## Summary 

- **Embeddings:** $$X^{(0)}=E[\mathbf{t}]+P \;\in\; [B,T,D]$$
- **Attn pre-LN:** $$\tilde{X}=\mathrm{LN}(X^{(\ell-1)}) \;\in\; [B,T,D]$$
- **Q/K/V:** $$Q=\tilde{X}W_Q,\; K=\tilde{X}W_K,\; V=\tilde{X}W_V \;\in\; [B,T,D]$$
- **Scores:** $$S=QK^{\top}/\sqrt{D}+M \;\in\; [B,T,T]$$
- **Weights:** $$A=\mathrm{softmax}(S) \;\in\; [B,T,T]$$
- **Context:** $$H=A\,V \;\in\; [B,T,D]$$
- **Proj:** $$O=H\,W_O \;\in\; [B,T,D]$$
- **Residual:** $$X'=X^{(\ell-1)}+O \;\in\; [B,T,D]$$
- **MLP pre-LN:** $$\hat{X}=\mathrm{LN}(X') \;\in\; [B,T,D]$$
- **MLP:** $$G=\mathrm{GELU}(\hat{X}W_1+b_1) \;\in\; [B,T,d_{\text{ff}}]$$
- **MLP proj:** $$M=G\,W_2+b_2 \;\in\; [B,T,D]$$
- **Residual:** $$X^{(\ell)}=X'+M \;\in\; [B,T,D]$$
- **Final:** $$X_f=\mathrm{LN}_f(X^{(L)}) \;\in\; [B,T,D]$$
- **Logits:** $$Z=X_fE^{\top}$$ (tied) **or** $$Z=X_fW_{\text{vocab}}$$ (untied) $$\;\in\; [B,T,V]$$

---

# Part II: ViT → Text (Single-Stream)

Building on the nanoGPT foundation, we can now extend to Vision Transformers that process both image and text data in a unified architecture. The primary difference is that we'd now like to take our image, patchify it and map that patches into tokens so we can have the text and image data in the same format and domain to be processed by our attention layers. 

We also assume that we want text output. 

## Notation / sizes (ViT Extension)

- Batch $$B$$; image $$(H\times W)$$ with channels $$C$$
- Patch size $$P$$; number of patches $$N=\frac{H}{P}\cdot\frac{W}{P}$$
- Text length $$T$$, vocab size $$V$$
- Model width $$D$$, MLP hidden width $$d_{\text{ff}}$$ (e.g., $$4D$$)
- Single head so $$d_k=d_v=D$$ (keeps shapes simple)
- Pre-LayerNorm residual layout (GPT-style)

---

## 0) Inputs → tokens

### 0.1 Image → patch tokens

Define an **unfold** operator $$\mathcal{U}$$ that extracts flattened non-overlapping patches:
$$\mathcal{U}:\ \mathbb{R}^{B\times C\times H\times W}\ \to\ \mathbb{R}^{B\times N\times (C P^2)}$$

Let the image batch be $$X_{\text{img}}\in\mathbb{R}^{B\times C\times H\times W}$$. Then
$$X_{\text{patch}}=\mathcal{U}(X_{\text{img}})\ \in\ \mathbb{R}^{B\times N\times (C P^2)}$$

Linear patch projection (ViT style):
$$W_{\text{patch}}\in\mathbb{R}^{(C P^2)\times D},\quad b_{\text{patch}}\in\mathbb{R}^{D},\qquad I=X_{\text{patch}}\,W_{\text{patch}}+b_{\text{patch}}\ \in\ \mathbb{R}^{B\times N\times D}$$

(Equivalently: a Conv2d with kernel$$=$$stride$$=P$$ giving $$[B,D,H/P,W/P]$$, then flatten to $$[B,N,D]$$.)

Add **2D positional** embeddings for image patches (flattened scan order):
$$P_{\text{img}}\in\mathbb{R}^{N\times D},\qquad I^{(0)}=I+P_{\text{img}}\ \in\ \mathbb{R}^{B\times N\times D}$$

(Optional **type**/modality embedding $$T_{\text{img}}\in\mathbb{R}^{D}$$: add via broadcast if desired.)

### 0.2 Text → token embeddings

Token ids: $$t\in\{1,\dots,V\}^{B\times T}$$.  
Embedding matrix: $$E\in\mathbb{R}^{V\times D}$$.  
1D positional embeddings: $$P_{\text{txt}}\in\mathbb{R}^{T\times D}$$.
$$X_{\text{txt}}^{(0)} = E[t] + P_{\text{txt}} \ \in\ \mathbb{R}^{B\times T\times D}$$

(Optional type embedding $$T_{\text{txt}}\in\mathbb{R}^{D}$$: add via broadcast.)

### 0.3 Concatenate image+text as one sequence

Place **image tokens first** so text can attend to them for generation:
$$X^{(0)}=\operatorname{concat}\!\big(I^{(0)},\ X_{\text{txt}}^{(0)}\big)\ \in\ \mathbb{R}^{B\times S\times D},\quad S=N+T$$

---

## 1) Transformer blocks $$\ell=1,\dots,L$$ (single head)

### 1.1 Pre-norm (attention)

$$\tilde{X}=\mathrm{LN}^{(\ell)}_{\text{attn}}\!\left(X^{(\ell-1)}\right)\ \in\ \mathbb{R}^{B\times S\times D}$$

### 1.2 Q/K/V projections

$$W_Q^{(\ell)},\ W_K^{(\ell)},\ W_V^{(\ell)}\ \in\ \mathbb{R}^{D\times D}$$

$$Q=\tilde{X}W_Q^{(\ell)},\quad K=\tilde{X}W_K^{(\ell)},\quad V=\tilde{X}W_V^{(\ell)}\ \in\ \mathbb{R}^{B\times S\times D}$$

### 1.3 Scaled dot-product attention with **causal** mask

Scores:
$$S=\frac{QK^{\top}}{\sqrt{D}}+M\ \in\ \mathbb{R}^{B\times S\times S}$$

where $$M_{ij}=0$$ if $$j\le i$$ and $$-\infty$$ otherwise (standard upper-triangular mask).

- Image tokens occupy positions $$1\ldots N$$ and cannot see future text tokens.
- Text tokens (positions $$N\!+\!1\ldots N\!+\!T$$) can attend to **all** image tokens and prior text tokens.

Row-wise softmax and context:
$$A=\mathrm{softmax}(S)\in\mathbb{R}^{B\times S\times S},\qquad H=AV\in\mathbb{R}^{B\times S\times D}$$

Output projection and residual:
$$W_O^{(\ell)}\in\mathbb{R}^{D\times D},\quad O=HW_O^{(\ell)}\in\mathbb{R}^{B\times S\times D},\quad X'=X^{(\ell-1)}+O$$

### 1.4 Pre-norm (MLP)

$$\hat{X}=\mathrm{LN}^{(\ell)}_{\text{mlp}}(X')\in\mathbb{R}^{B\times S\times D}$$

### 1.5 MLP + GELU

$$W_1^{(\ell)}\in\mathbb{R}^{D\times d_{\text{ff}}},\quad W_2^{(\ell)}\in\mathbb{R}^{d_{\text{ff}}\times D}$$

$$U=\hat{X}W_1^{(\ell)}+b_1^{(\ell)}\in\mathbb{R}^{B\times S\times d_{\text{ff}}},\quad G=\mathrm{GELU}(U)\in\mathbb{R}^{B\times S\times d_{\text{ff}}}$$

$$M=GW_2^{(\ell)}+b_2^{(\ell)}\in\mathbb{R}^{B\times S\times D},\quad X^{(\ell)}=X'+M\in\mathbb{R}^{B\times S\times D}$$

Repeat for $$\ell=1,\dots,L$$.

---

## 2) Output head → **text logits**

Final LayerNorm:
$$X_f=\mathrm{LN}_f\!\left(X^{(L)}\right)\in\mathbb{R}^{B\times S\times D}$$

Select only the **text segment** (positions $$N+1\ldots N+T$$):
$$X_{f,\text{txt}}=X_f[:,\,N:\!,:]\in\mathbb{R}^{B\times T\times D}$$

Two head options:

**(a) Weight tying (reuse text embeddings $$E\in\mathbb{R}^{V\times D}$$):**
$$Z=X_{f,\text{txt}}\,E^\top\ \in\ \mathbb{R}^{B\times T\times V}$$

**(b) Separate head:**
$$Z=X_{f,\text{txt}}\,W_{\text{vocab}} + b_{\text{vocab}},\qquad W_{\text{vocab}}\in\mathbb{R}^{D\times V}$$

These $$Z$$ are **text logits**; training uses cross-entropy on the text tokens (shifted by one), masking out any image positions as needed.

---

## Summary 

- **Image patches:** $$X_{\text{patch}}=\mathcal{U}(X_{\text{img}}) \;\in\; [B,N,CP^2]$$
- **Patch embed:** $$I=X_{\text{patch}}W_{\text{patch}}+b_{\text{patch}} \;\in\; [B,N,D]$$
- **Image pos:** $$I^{(0)}=I+P_{\text{img}} \;\in\; [B,N,D]$$
- **Text embed:** $$X_{\text{txt}}^{(0)}=E[t]+P_{\text{txt}} \;\in\; [B,T,D]$$
- **Concatenate:** $$X^{(0)}=\operatorname{concat}(I^{(0)},X_{\text{txt}}^{(0)}) \;\in\; [B,S,D], \; S=N+T$$
- **Pre-norm:** $$\tilde{X}=\mathrm{LN}(X^{(\ell-1)}) \;\in\; [B,S,D]$$
- **Q/K/V:** $$Q,K,V=\tilde{X}W_Q,\,\tilde{X}W_K,\,\tilde{X}W_V \;\in\; [B,S,D]$$
- **Scores:** $$S=QK^\top/\sqrt{D}+M \;\in\; [B,S,S]$$
- **Attention:** $$A=\mathrm{softmax}(S) \;\in\; [B,S,S]$$
- **Values:** $$H=AV \;\in\; [B,S,D]$$
- **Output proj:** $$O=HW_O \;\in\; [B,S,D]$$
- **Add & norm:** $$X'=X^{(\ell-1)}+O \;\in\; [B,S,D]$$
- **Pre-norm MLP:** $$\hat{X}=\mathrm{LN}(X') \;\in\; [B,S,D]$$
- **MLP up:** $$U=\hat{X}W_1+b_1 \;\in\; [B,S,d_{\text{ff}}]$$
- **GELU:** $$G=\mathrm{GELU}(U) \;\in\; [B,S,d_{\text{ff}}]$$
- **MLP down:** $$M=GW_2+b_2 \;\in\; [B,S,D]$$
- **Add:** $$X^{(\ell)}=X'+M \;\in\; [B,S,D]$$
- **Final norm:** $$X_f=\mathrm{LN}_f(X^{(L)}) \;\in\; [B,S,D]$$
- **Text only:** $$X_{f,\text{txt}}=X_f[:,N:,:] \;\in\; [B,T,D]$$
- **Logits:** $$Z=X_{f,\text{txt}}E^\top$$ (tied) **or** $$Z=X_{f,\text{txt}}W_{\text{vocab}}$$ (untied) $$\;\in\; [B,T,V]$$


I hope this is helpful in understanding the dataflow through a transformer block for images and text token. 

---

