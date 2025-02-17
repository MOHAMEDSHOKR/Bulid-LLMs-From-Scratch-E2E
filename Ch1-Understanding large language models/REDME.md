# Introduction ***
LLMs have remarkable capabilities to understand, generate, and interpret human
language. However, it’s important to clarify that when we say language models “understand,”
we mean that they can process and generate text in ways that appear coherent
and contextually relevant, not that they possess human-like consciousness or
comprehension.
Enabled by advancements in deep learning, which is a subset of machine learning
and *** artificial intelligence (AI) *** focused on neural networks, LLMs are trained on
vast quantities of text data. This large-scale training allows LLMs to capture deeper
contextual information and subtleties of human language compared to previous
approaches. As a result, LLMs have significantly improved performance in a wide
range of NLP tasks, including text translation, sentiment analysis, question answering,
and many more.
Another important distinction between contemporary LLMs and earlier NLP models
is that earlier NLP models were typically designed for specific tasks, such as text
categorization, language translation, etc. While those earlier NLP models excelled in
their narrow applications, LLMs demonstrate a broader proficiency across a wide
range of NLP tasks.
The success behind LLMs can be attributed to the transformer architecture that
underpins many LLMs and the vast amounts of data on which LLMs are trained,
allowing them to capture a wide variety of linguistic nuances, contexts, and patterns
that would be challenging to encode manually.
This shift toward implementing models based on the transformer architecture and
using large training datasets to train LLMs has fundamentally transformed NLP, providing
more capable tools for understanding and interacting with human language.
The following discussion sets a foundation to accomplish the primary objective of
this book: understanding LLMs by implementing a ChatGPT-like LLM based on the
transformer architecture step by step in code.

# 1.1 What is an LLM?
An LLM is a neural network designed to understand, generate, and respond to humanlike
text. These models are deep neural networks trained on massive amounts of text
data, sometimes encompassing large portions of the entire publicly available text on
the internet.
The “large” in “large language model” refers to both the model’s size in terms of
parameters and the immense dataset on which it’s trained. Models like this often have
tens or even hundreds of billions of parameters, which are the adjustable weights in
the network that are optimized during training to predict the next word in a sequence.
Next-word prediction is sensible because it harnesses the inherent sequential nature
of language to train models on understanding context, structure, and relationships
within text. Yet, it is a very simple task, and so it is surprising to many researchers that
it can produce such capable models. In later chapters, we will discuss and implement
the next-word training procedure step by step.
