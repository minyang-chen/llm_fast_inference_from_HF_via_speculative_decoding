## Problem
Large autoregressive models, notably large Transformers (Vaswani et al., 2017), are much more capable than smaller models, as is evidenced countless times in recent year.
Unfortunately, a single decode step from these larger models is significantly slower than a step from their smaller counterparts, and making things worse, these steps are done serially - decoding K tokens takes K se>

## What are the problems with quantization?
While quantized LLM models offer various benefits such as reduced memory footprint and improved computational efficiency, there are some potential disadvantages to consider: Decreased Model Accuracy: Quantization involves reducing the precision of numerical values in the model, which can lead to a loss of information

## llm Fast Inference from Transformers via Speculative Decoding
Google Research publish an interesting paper (Fast Inference from Transformers via Speculative Decoding)[https://browse.arxiv.org/pdf/2211.17192.pdf] that promising 2-3X speedups of LLM inference by running two models in parallel. The core idea is using a faster, and lower quality model, that approximates the target model to sample multiple tokens and then check these samples using the target model.  

E.g. Sample from TinyLlama quickly, then use LLaMA 13b to check the samples.

## Motivation
Adding this kind of support to Inference would make LLM sampling much faster without change the output.



