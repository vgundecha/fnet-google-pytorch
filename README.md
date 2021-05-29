# fnet-google-pytorch
Pytorch implementation of [FNet: Mixing Tokens with Fourier Transforms by Google Research](https://arxiv.org/pdf/2105.03824.pdf)

This paper replace the Self-Attention block in the Transformer architecture with a Fourier Transform

<img src="./fnet.png" width="300px"></img>

## Basic idea
 - Attention is a mechanism for facilitating interaction between tokens in a sequence (mixing tokens)
 - This paper mixes tokens using Discrete Fourier Transform.
 - Advantage: 
   - DFT is unparametrized (the basis/weights are fixed, drastic reduction in num. of params.)
   - Computing DFT using FFT is extremely fast.
   
## Usage
```python
 import torch
 from fnet import Fnet
 # N = number of layers, dhidden = input embedding size
 model = Fnet(N=2, dhidden=32)
 model = model.train(False)
 # Input embedding (not included in the model): batch_size=2, sequence_length=8, dhidden=32
 x = torch.randn((2, 8, 32))
 y = model(x) # y.shape = (2, 8, 32), model representation without the output projection
 ```
 
 ## Citation
 ```bibtex
 @misc{leethorp2021fnet,
      title={FNet: Mixing Tokens with Fourier Transforms}, 
      author={James Lee-Thorp and Joshua Ainslie and Ilya Eckstein and Santiago Ontanon},
      year={2021},
      eprint={2105.03824},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
