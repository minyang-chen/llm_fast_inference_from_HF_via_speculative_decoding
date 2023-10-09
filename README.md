## Medium
https://mychen76.medium.com/evaluate-llm-fast-inference-via-speculative-decoding-without-quantization-f2cfbb32e33c

## Context
Large autoregressive models, notably large Transformers (Vaswani et al., 2017), are much more capable than smaller models, as is evidenced countless times in recent year.
Unfortunately, a single decode step from these larger models is significantly slower than a step from their smaller counterparts, and making things worse, these steps are done serially - decoding K tokens takes K se>

## What are the problems with quantization?
While quantized LLM models offer various benefits such as reduced memory footprint and improved computational efficiency, there are some potential disadvantages to consider: Decreased Model Accuracy: Quantization involves reducing the precision of numerical values in the model, which can lead to a loss of information

## llm Fast Inference from Transformers via Speculative Decoding
Google Research publish an interesting paper (Fast Inference from Transformers via Speculative Decoding)[https://browse.arxiv.org/pdf/2211.17192.pdf] that promising 2-3X speedups of LLM inference by running two models in parallel. The core idea is using a faster, and lower quality model, that approximates the target model to sample multiple tokens and then check these samples using the target model.  

E.g. Sample from TinyLlama quickly, then use LLaMA 13b to check the samples.

## Motivation
Adding speculative deconding support to Inference would make LLM sampling much faster without change the output.

## Test Prompts
test_prompt_list = [
```
"Name planets in our solar system",        
"Give me detailed info about Justin Trudeau.",                 
"Generate a few good titles for a draft of a post [type of transformer models]",
"Write a 5-line poem that describes a cat in a creative and original way. Use the following words in your poem: cat, fur, tail, independent, and clean.",
"Given the text below, generate Q&A from the text provide: [rewrite of minGPT that prioritizes teeth over education.]",
"Write a highly detailed discussion response, in the structure of an essay, responding to the following prompt: Explain the causes of the Inflation and whether expansion played a role in the economic recession. Include evidence to support your argument.",
"Make a coding challenge for Python with 3 questions and answers"
```
]

## Environment Setup and Requirement 
-- PC with at least 32 GB Ram
-- GPU with 24 GB VRAM

## Run Test 
```
# if required
pip install -r requirements.txt

# run inference test
python speculative_decoding_evaluation.py
```

## Test Result (in seconds)
Native Decoding only:
```
['68.12223672866821', '55.81372261047363', '20.210347890853882', '28.41849422454834', '72.30130910873413', '68.66965794563293', '69.7259030342102']
```
With Assisted Speculate Decoding :
```
['8.641464233398438', '38.08631610870361', '21.30632519721985', '8.92543363571167', '40.273754358291626', '38.650553941726685', '37.184149503707886']
```
## Observation 
Overall walltime suggest running Big and Small model in paralle -- Assisted Speculate Decoding reduce inference time at least by half.

