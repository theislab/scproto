# Interpretable Self-Supervised Prototype Learning for Single-Cell Transcriptomics

**Learn cross-batch metacells** using interpretable self-supervised prototype learning to **denoise** and **preserve biological structure** in single-cell data — without using labels.

📄 [Read the paper (ICLR 2025 - LMRL Workshop)](https://openreview.net/forum?id=mTjWUeyll5&noteId=mTjWUeyll5)


## Overview

Single-cell RNA-seq data is often **noisy**, **sparse**, and affected by **batch effects**, which can obscure meaningful biological insights.  
**scProto** is an interpretable self-supervised prototype learning method that learns biologically meaningful **prototypes** and decodes them into **metacells** — compact, denoised representations of cell populations across batches.

- Learns **cross-batch metacells** that reflect biologically meaningful cell groups  
- Trained to **preserve biological structure** and **cell-cell relationships** in the embedding space while mitigating batch effects  
- Fully **label-free**, requiring no annotations  

## Key Features

- **Interpretable prototype learning** and **metacell decoding** across datasets  
- Embedding space that maintains **biological topology** and **local cell relationships**  
- Enhances single-cell analysis by **denoising gene expression** and **overcoming data sparsity**
