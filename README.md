# Comparison of Autoencoder Methods: CAE, VAE and VQ-VAE for Data Reconstruction

This repository contains an experimental comparison of three autoencoder architectures:

- **Contractive Auto-Encoder (CAE)**
- **Variational Auto-Encoder (VAE)**
- **Vector-Quantized Variational Auto-Encoder (VQ-VAE)**

The main goal is to study how these models differ in terms of:

- **Reconstruction quality**
- **Latent space structure and properties**
- **Robustness and stability during training**

---

## Project Overview

Autoencoders are composed of two main components:

- **Encoder** – compresses input data into a lower-dimensional **latent representation**  
- **Decoder** – reconstructs the original input from this latent representation  

In this project we compare three different approaches to building the latent space:

1. **CAE (Contractive Auto-Encoder)**  
   - Adds a contractive penalty (Frobenius norm of the Jacobian of the encoder) to the loss  
   - Encourages **local invariance** and robustness to small perturbations of the input  

2. **VAE (Variational Auto-Encoder)**  
   - Treats the latent variables as **probabilistic**  
   - Learns a distribution in the latent space and optimizes the **ELBO**  
   - Enables **generative modeling** and sampling from the latent space  

3. **VQ-VAE (Vector-Quantized VAE)**  
   - Replaces continuous latent variables with a **discrete codebook**  
   - Produces symbolic latent representations  
   - Latent codes can be further modeled by **autoregressive models** or **transformers**  

We hypothesize that **VQ-VAE** will achieve the **best reconstruction quality**, thanks to the codebook-based representation, while CAE and VAE will offer different trade-offs between smoothness, interpretability and generative properties.

---

## Key Objectives

- Implement CAE, VAE and VQ-VAE in **PyTorch**
- Train all models on a **shared dataset**
- Compare:
  - Reconstruction quality (quantitative metrics and qualitative visualizations)
  - Latent space structure (e.g. projections / clustering)
  - Training behavior and stability  

---

## Technologies

This project uses:

- **Python**
- **PyTorch** – model implementation and training
- **NumPy** – numerical operations
- **Matplotlib** – plotting and visualization
- **scikit-learn** – auxiliary analysis (e.g. dimensionality reduction, clustering)

