---
title: 'ViT Transformation Notation'
date: 2025-01-15
permalink: /posts/2025/01/vit-transformation-notation/
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

This is a comprehensive notation primer on transformers and Vision Transformers (ViTs). We'll start with the mathematical foundations of nanoGPT to establish core transformer concepts, then extend to ViT architectures that process both image and text data.

---

# Part I: NanoGPT Foundation

This section covers the mathematical notation for nanoGPT, providing the foundational understanding of transformer architectures.

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

## NanoGPT Summary (shapes inline)

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

Building on the nanoGPT foundation, we now extend to Vision Transformers that process both image and text data in a unified architecture.

## Notation / sizes (ViT Extension)

- Batch $$B$$; image $$(H\times W)$$ with channels $$C$$
- Patch size $$P$$ (assume $$P\mid H,\,P\mid W$$); number of patches $$N=\frac{H}{P}\cdot\frac{W}{P}$$
- Text length $$T$$, vocab size $$V$$
- Model width $$D$$, MLP hidden width $$d_{\text{ff}}$$ (e.g., $$4D$$)
- **Single head** so $$d_k=d_v=D$$ (keeps shapes simple)
- **Pre-LayerNorm** residual layout (GPT-style)

---

## 0) Inputs → tokens

### 0.1 Image → patch tokens

Define an **unfold** operator $$\mathcal{U}$$ that extracts flattened non-overlapping patches:
$$\mathcal{U}:\ \mathbb{R}^{B\times C\times H\times W}\ \to\ \mathbb{R}^{B\times N\times (C P^2)}$$

Let the image batch be $$\mathbf{X}_{\text{img}}\in\mathbb{R}^{B\times C\times H\times W}$$. Then
$$\mathbf{X}_{\text{patch}}=\mathcal{U}(\mathbf{X}_{\text{img}})\ \in\ \mathbb{R}^{B\times N\times (C P^2)}$$

Linear patch projection (ViT style):
$$W_{\text{patch}}\in\mathbb{R}^{(C P^2)\times D},\quad b_{\text{patch}}\in\mathbb{R}^{D},\qquad \mathbf{I}=\mathbf{X}_{\text{patch}}\,W_{\text{patch}}+b_{\text{patch}}\ \in\ \mathbb{R}^{B\times N\times D}$$

(Equivalently: a Conv2d with kernel$$=$$stride$$=P$$ giving $$[B,D,H/P,W/P]$$, then flatten to $$[B,N,D]$$.)

Add **2D positional** embeddings for image patches (flattened scan order):
$$P_{\text{img}}\in\mathbb{R}^{N\times D},\qquad \mathbf{I}^{(0)}=\mathbf{I}+P_{\text{img}}\ \in\ \mathbb{R}^{B\times N\times D}$$

(Optional **type**/modality embedding $$T_{\text{img}}\in\mathbb{R}^{D}$$: add via broadcast if desired.)

### 0.2 Text → token embeddings

Token ids: $$\mathbf{t}\in\{1,\dots,V\}^{B\times T}$$.  
Embedding matrix: $$E\in\mathbb{R}^{V\times D}$$.  
1D positional embeddings: $$P_{\text{txt}}\in\mathbb{R}^{T\times D}$$.
$$\mathbf{X}_{\text{txt}}^{(0)} = E[\mathbf{t}] + P_{\text{txt}} \ \in\ \mathbb{R}^{B\times T\times D}$$

(Optional type embedding $$T_{\text{txt}}\in\mathbb{R}^{D}$$: add via broadcast.)

### 0.3 Concatenate image+text as one sequence

Place **image tokens first** so text can attend to them for generation:
$$\mathbf{X}^{(0)}=\operatorname{concat}\!\big(\mathbf{I}^{(0)},\ \mathbf{X}_{\text{txt}}^{(0)}\big)\ \in\ \mathbb{R}^{B\times S\times D},\quad S=N+T$$

---

## 1) Transformer blocks $$\ell=1,\dots,L$$ (single head)

### 1.1 Pre-norm (attention)

$$\tilde{\mathbf{X}}=\mathrm{LN}^{(\ell)}_{\text{attn}}\!\left(\mathbf{X}^{(\ell-1)}\right)\ \in\ \mathbb{R}^{B\times S\times D}$$

### 1.2 Q/K/V projections

$$W_Q^{(\ell)},\ W_K^{(\ell)},\ W_V^{(\ell)}\ \in\ \mathbb{R}^{D\times D}$$

$$\mathbf{Q}=\tilde{\mathbf{X}}W_Q^{(\ell)},\quad \mathbf{K}=\tilde{\mathbf{X}}W_K^{(\ell)},\quad \mathbf{V}=\tilde{\mathbf{X}}W_V^{(\ell)}\ \in\ \mathbb{R}^{B\times S\times D}$$

### 1.3 Scaled dot-product attention with **causal** mask

Scores:
$$\mathbf{S}=\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{D}}+\mathbf{M}\ \in\ \mathbb{R}^{B\times S\times S}$$

where $$\mathbf{M}_{ij}=0$$ if $$j\le i$$ and $$-\infty$$ otherwise (standard upper-triangular mask).

- Image tokens occupy positions $$1\ldots N$$ and cannot see future text tokens.
- Text tokens (positions $$N\!+\!1\ldots N\!+\!T$$) can attend to **all** image tokens and prior text tokens.

Row-wise softmax and context:
$$\mathbf{A}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{B\times S\times S},\qquad \mathbf{H}=\mathbf{A}\mathbf{V}\in\mathbb{R}^{B\times S\times D}$$

Output projection and residual:
$$W_O^{(\ell)}\in\mathbb{R}^{D\times D},\quad \mathbf{O}=\mathbf{H}W_O^{(\ell)}\in\mathbb{R}^{B\times S\times D},\quad \mathbf{X}'=\mathbf{X}^{(\ell-1)}+\mathbf{O}$$

### 1.4 Pre-norm (MLP)

$$\hat{\mathbf{X}}=\mathrm{LN}^{(\ell)}_{\text{mlp}}(\mathbf{X}')\in\mathbb{R}^{B\times S\times D}$$

### 1.5 MLP + GELU

$$W_1^{(\ell)}\in\mathbb{R}^{D\times d_{\text{ff}}},\quad W_2^{(\ell)}\in\mathbb{R}^{d_{\text{ff}}\times D}$$

$$\mathbf{U}=\hat{\mathbf{X}}W_1^{(\ell)}+b_1^{(\ell)}\in\mathbb{R}^{B\times S\times d_{\text{ff}}},\quad \mathbf{G}=\mathrm{GELU}(\mathbf{U})\in\mathbb{R}^{B\times S\times d_{\text{ff}}}$$

$$\mathbf{M}=\mathbf{G}W_2^{(\ell)}+b_2^{(\ell)}\in\mathbb{R}^{B\times S\times D},\quad \mathbf{X}^{(\ell)}=\mathbf{X}'+\mathbf{M}\in\mathbb{R}^{B\times S\times D}$$

Repeat for $$\ell=1,\dots,L$$.

---

## 2) Output head → **text logits**

Final LayerNorm:
$$\mathbf{X}_f=\mathrm{LN}_f\!\left(\mathbf{X}^{(L)}\right)\in\mathbb{R}^{B\times S\times D}$$

Select only the **text segment** (positions $$N+1\ldots N+T$$):
$$\mathbf{X}_{f,\text{txt}}=\mathbf{X}_f[:,\,N:\!,:]\in\mathbb{R}^{B\times T\times D}$$

Two head options:

**(a) Weight tying (reuse text embeddings $$E\in\mathbb{R}^{V\times D}$$):**
$$\mathbf{Z}=\mathbf{X}_{f,\text{txt}}\,E^\top\ \in\ \mathbb{R}^{B\times T\times V}$$

**(b) Separate head:**
$$\mathbf{Z}=\mathbf{X}_{f,\text{txt}}\,W_{\text{vocab}} + b_{\text{vocab}},\qquad W_{\text{vocab}}\in\mathbb{R}^{D\times V}$$

These $$\mathbf{Z}$$ are **text logits**; training uses cross-entropy on the text tokens (shifted by one), masking out any image positions as needed.

---

## One-page summary (shapes inline)

**Image patchify & embed**
$$\mathbf{X}_{\text{patch}}=\mathcal{U}(\mathbf{X}_{\text{img}})\ \in\ [B,N,CP^2],\quad \mathbf{I}=\mathbf{X}_{\text{patch}}W_{\text{patch}}+b_{\text{patch}}\ \in\ [B,N,D],\quad \mathbf{I}^{(0)}=\mathbf{I}+P_{\text{img}}\ \in\ [B,N,D]$$

**Text tokenize**
$$\mathbf{X}_{\text{txt}}^{(0)}=E[\mathbf{t}]+P_{\text{txt}}\ \in\ [B,T,D]$$

**Concatenate**
$$\mathbf{X}^{(0)}=\operatorname{concat}(\mathbf{I}^{(0)},\mathbf{X}_{\text{txt}}^{(0)})\ \in\ [B,S,D],\ S=N+T$$

**Block (single head)**
$$\tilde{\mathbf{X}}=\mathrm{LN}(\mathbf{X}^{(\ell-1)})\ \in\ [B,S,D],\quad \mathbf{Q},\mathbf{K},\mathbf{V}=\tilde{\mathbf{X}}W_Q,\,\tilde{\mathbf{X}}W_K,\,\tilde{\mathbf{X}}W_V\ \in\ [B,S,D]$$

$$\mathbf{S}=\mathbf{Q}\mathbf{K}^\top/\sqrt{D}+\mathbf{M}\ \in\ [B,S,S],\quad \mathbf{A}=\mathrm{softmax}(\mathbf{S})\ \in\ [B,S,S],\quad \mathbf{H}=\mathbf{A}\mathbf{V}\ \in\ [B,S,D]$$

$$\mathbf{O}=\mathbf{H}W_O\ \in\ [B,S,D],\quad \mathbf{X}'=\mathbf{X}^{(\ell-1)}+\mathbf{O}\ \in\ [B,S,D]$$

$$\hat{\mathbf{X}}=\mathrm{LN}(\mathbf{X}')\ \in\ [B,S,D],\quad \mathbf{G}=\mathrm{GELU}(\hat{\mathbf{X}}W_1+b_1)\ \in\ [B,S,d_{\text{ff}}],\quad \mathbf{M}=\mathbf{G}W_2+b_2\ \in\ [B,S,D]$$

$$\mathbf{X}^{(\ell)}=\mathbf{X}'+\mathbf{M}\ \in\ [B,S,D]$$

**Head (text only)**
$$\mathbf{X}_f=\mathrm{LN}_f(\mathbf{X}^{(L)})\ \in\ [B,S,D],\quad \mathbf{X}_{f,\text{txt}}=\mathbf{X}_f[:,N:\!,:]\ \in\ [B,T,D]$$

$$\mathbf{Z}=\mathbf{X}_{f,\text{txt}}E^\top\ \text{ (tied) }\quad \text{or}\quad \mathbf{Z}=\mathbf{X}_{f,\text{txt}}W_{\text{vocab}} \ \text{ (untied) }\ \in\ [B,T,V]$$

---

## Notes / variants

- **Bidirectional vision:** To allow all image patches to see each other, set the top-left $$N\times N$$ block of $$\mathbf{M}$$ to zeros (no causal masking there), keep causal masking for the bottom-right $$T\times T$$ text block, and disallow image$$\to$$text attention by setting the top-right block to $$-\infty$$. Text$$\to$$image (bottom-left) remains allowed.
- **Special tokens:** You may prepend a learned $$[\text{IMG}]$$ or delimiter tokens; include them in the concatenation and in positional indices.
- **Rotary/relative pos:** Replace $$P_{\text{img}},P_{\text{txt}}$$ with RoPE/relative encodings; shapes remain unchanged.
- **Dim mismatch:** If text or vision embeddings are not $$D$$-dimensional, insert a linear projection to $$D$$ before concatenation.

## Summary

This comprehensive notation primer provides:

1. **NanoGPT Foundation**: Core transformer concepts including self-attention, causal masking, pre-LayerNorm architecture, and autoregressive text generation
2. **ViT Extension**: Vision-text multimodal architectures that combine image patch processing with text generation in a unified transformer framework

Together, these mathematical frameworks enable understanding and implementing modern transformer architectures from simple language models to complex multimodal systems, maintaining precise tensor dimensions and operations throughout both pipelines. 