
# This chapter covers
### *** High-level explanations of the fundamental concepts behind large language models (LLMs)
### *** Insights into the transformer architecture from which LLMs are derived
### *** A plan for building an LLM from scratch

Large language models (LLMs), such as those offered in OpenAI’s ChatGPT, are
deep neural network models that have been developed over the past few years.
They ushered in a new era for natural language processing (NLP). Before the
advent of LLMs, traditional methods excelled at categorization tasks such as email
spam classification and straightforward pattern recognition that could be captured
with handcrafted rules or simpler models. However, they typically underperformed
in language tasks that demanded complex understanding and generation abilities,
such as parsing detailed instructions, conducting contextual analysis, and creating
coherent and contextually appropriate original text. For example, previous generations
of language models could not write an email from a list of keywords—a task
that is trivial for contemporary LLMs.

### *** LLMs have remarkable capabilities to understand, generate, and interpret human language.
However, it’s important to clarify that when we say language models “understand,”
we mean that they can process and generate text in ways that appear coherent
and contextually relevant, not that they possess human-like consciousness or
comprehension.
Enabled by advancements in deep learning, which is a subset of machine learning
and artificial intelligence (AI) focused on neural networks, LLMs are trained on
vast quantities of text data. This large-scale training allows LLMs to capture deeper
contextual information and subtleties of human language compared to previous
approaches. As a result, LLMs have significantly improved performance in a wide
range of NLP tasks, including text translation, sentiment analysis, question answering,
and many more.
### *** Another important distinction between contemporary LLMs and earlier NLP models
is that earlier NLP models were typically designed for specific tasks, such as text
categorization, language translation, etc. While those earlier NLP models excelled in
their narrow applications, LLMs demonstrate a broader proficiency across a wide
range of NLP tasks.

### *** The success behind LLMs can be attributed to the transformer architecture that
underpins many LLMs and the vast amounts of data on which LLMs are trained,
allowing them to capture a wide variety of linguistic nuances, contexts, and patterns
that would be challenging to encode manually.
This shift toward implementing models based on the transformer architecture and
using large training datasets to train LLMs has fundamentally transformed NLP, providing
more capable tools for understanding and interacting with human language.
The following discussion sets a foundation to accomplish the primary objective of
this book: understanding LLMs by implementing a ChatGPT-like LLM based on the
transformer architecture step by step in code.
