# LLMs-From-Scratch
### *** Building a Large Language Model (From Scratch) *** to understand and create your own GPT-like large language models (LLMs) from the ground up.
It begins by focusing on the fundamentals of working with text data and coding attention mechanisms 
and then guides you through implementing a complete GPT model from scratch.
The book then covers the pretraining mechanism as well as
fine-tuning for specific tasks such as text classification and following instructions. 
By the end of this book, you’ll have a deep understanding of how LLMs work and the skills to build your own models.
While the models you’ll create are smaller in scale compared to the large foundational models, 
they use the same concepts and serve as powerful educational tools to grasp the core mechanisms and techniques
used in building state-of-the-art LLMs.

# How this book is organized: A roadmap
by using LLMs from Scratch book is designed to be read sequentially, as each chapter builds upon the concepts
and techniques introduced in the previous ones. The book is divided into seven
chapters that cover the essential aspects of LLMs and their implementation.

# Chapter 1  provides a high-level introduction to the fundamental concepts behind
LLMs. It explores the transformer architecture, which forms the basis for LLMs such
as those used on the ChatGPT platform.
# Chapter 2 lays out a plan for building an LLM from scratch. It covers the process of
preparing text for LLM training, including splitting text into word and subword
tokens, using byte pair encoding for advanced tokenization, sampling training examples
with a sliding window approach, and converting tokens into vectors that feed into
the LLM.
# Chapter 3 focuses on the attention mechanisms used in LLMs. It introduces a basic
self-attention framework and progresses to an enhanced self-attention mechanism.
The chapter also covers the implementation of a causal attention module that enables
LLMs to generate one token at a time, masking randomly selected attention weights
with dropout to reduce overfitting and stacking multiple causal attention modules
into a multihead attention module.
# Chapter 4 focuses on coding a GPT-like LLM that can be trained to generate
human-like text. It covers techniques such as normalizing layer activations to stabilize
neural network training, adding shortcut connections in deep neural networks to
train models more effectively, implementing transformer blocks to create GPT models
of various sizes, and computing the number of parameters and storage requirements
of GPT models.
# Chapter 5 implements the pretraining process of LLMs. It covers computing the
training and validation set losses to assess the quality of LLM-generated text, implementing
a training function and pretraining the LLM, saving and loading model
weights to continue training an LLM, and loading pretrained weights from OpenAI.
# Chapter 6 introduces different LLM fine-tuning approaches. It covers preparing a
dataset for text classification, modifying a pretrained LLM for fine-tuning, fine-tuning
an LLM to identify spam messages, and evaluating the accuracy of a fine-tuned LLM
classifier.
# Chapter 7 explores the instruction fine-tuning process of LLMs. It covers preparing
a dataset for supervised instruction fine-tuning, organizing instruction data in
training batches, loading a pretrained LLM and fine-tuning it to follow human
instructions, extracting LLM-generated instruction responses for evaluation, and evaluating
an instruction-fine-tuned LLM.
