# 1.4 Introducing the transformer architecture
Most modern LLMs rely on the transformer architecture, which is a deep neural network
architecture introduced in the 2017 paper “Attention Is All You Need” (https://
arxiv.org/abs/1706.03762).

To understand LLMs, we must understand the original
transformer, which was developed for machine translation, translating English texts to
German and French. A simplified version of the transformer architecture is depicted
in figure 1.4.

The transformer architecture consists of two submodules: *** an encoder *** and
*** a decoder ***. The encoder module processes the input text and encodes it into a series of
numerical representations or vectors that capture the contextual information of the
input. 

Then, the decoder module takes these encoded vectors and generates the output
text. In a translation task, for example, the encoder would encode the text from
the source language into vectors, and the decoder would decode these vectors to generate
text in the target language. 

Both the encoder and decoder consist of many layers
connected by a so-called self-attention mechanism. You may have many questions
regarding how the inputs are preprocessed and encoded. These will be addressed in a
step-by-step implementation in subsequent chapters.

![image](https://github.com/user-attachments/assets/6d6903b4-1a6c-4ba9-a206-9b81f362a8b9)
Figure 1.4 A simplified depiction of the original transformer architecture, which is a deep learning model for
language translation. The transformer consists of two parts: (a) an encoder that processes the input text and
produces an embedding representation (a numerical representation that captures many different factors in
different dimensions) of the text that the (b) decoder can use to generate the translated text one word at a time.
This figure shows the final stage of the translation process where the decoder has to generate only the final word
(“Beispiel”), given the original input text (“This is an example”) and a partially translated sentence (“Das ist
ein”), to complete the translation.


# A key component of transformers and LLMs is the self-attention mechanism 
(not shown), which allows the model to weigh the importance of different words or tokens
in a sequence relative to each other. This mechanism enables the model to capture
long-range dependencies and contextual relationships within the input data, enhancing
its ability to generate coherent and contextually relevant output. However, due to its complexity,
we will defer further explanation to chapter 3, where we will discuss and implement it step by step.

Later variants of the transformer architecture, such as BERT (short for bidirectional
encoder representations from transformers) and the various GPT models (short for generative
pretrained transformers), built on this concept to adapt this architecture for different
tasks.

BERT, which is built upon the original transformer’s encoder submodule, differs
in its training approach from GPT. While GPT is designed for generative tasks, BERT
and its variants specialize in masked word prediction, where the model predicts masked
or hidden words in a given sentence, as shown in figure 1.5. This unique training strategy
equips BERT with strengths in text classification tasks, including sentiment prediction
and document categorization.

GPT, on the other hand, focuses on the decoder portion of the original transformer
architecture and is designed for tasks that require generating texts. This includes
machine translation, text summarization, fiction writing, writing computer code,
and more. 

![image](https://github.com/user-attachments/assets/91765a56-d669-42ac-8580-42af9d6ca6fd)
Figure 1.5 A visual representation of the transformer’s encoder and decoder submodules. On the left, the
encoder segment exemplifies BERT-like LLMs, which focus on masked word prediction and are primarily used for
tasks like text classification. On the right, the decoder segment showcases GPT-like LLMs, designed for
generative tasks and producing coherent text sequences.


GPT models, primarily designed and trained to perform text completion tasks,
also show remarkable versatility in their capabilities. These models are adept at executing
both zero-shot and few-shot learning tasks. Zero-shot learning refers to the ability
to generalize to completely unseen tasks without any prior specific examples. On
the other hand, few-shot learning involves learning from a minimal number of examples
the user provides as input, as shown in figure 1.6.

![image](https://github.com/user-attachments/assets/3cd788a6-36c8-49c1-8797-1211d1fb3684)
Figure 1.6 In addition to text completion, GPT-like LLMs can solve various tasks based on their inputs without
needing retraining, fine-tuning, or task-specific model architecture changes. Sometimes it is helpful to provide
examples of the target within the input, which is known as a few-shot setting. However, GPT-like LLMs are also
capable of carrying out tasks without a specific example, which is called zero-shot setting.


# Transformers vs. LLMs
Today’s LLMs are based on the transformer architecture. Hence, transformers and
LLMs are terms that are often used synonymously in the literature. However, note
that not all transformers are LLMs since transformers can also be used for computer
vision. Also, not all LLMs are transformers, as there are LLMs based on recurrent
and convolutional architectures. The main motivation behind these alternative
approaches is to improve the computational efficiency of LLMs. Whether these alternative
LLM architectures can compete with the capabilities of transformer-based
LLMs and whether they are going to be adopted in practice remains to be seen. For
simplicity, I use the term “LLM” to refer to transformer-based LLMs similar to GPT.


# 1.5 Utilizing large datasets

The large training datasets for popular GPT- and BERT-like models represent diverse
and comprehensive text corpora encompassing billions of words, which include a vast
array of topics and natural and computer languages. To provide a concrete example,
table 1.1 summarizes the dataset used for pretraining GPT-3, which served as the base
model for the first version of ChatGPT.

![image](https://github.com/user-attachments/assets/8c30ac28-0cf4-420d-b45d-d2a0988dcbfa)
Table 1.1 reports the number of tokens, where a token is a unit of text that a model
reads and the number of tokens in a dataset is roughly equivalent to the number of
words and punctuation characters in the text. Chapter 2 addresses tokenization, the
process of converting text into tokens.
The main takeaway is that the scale and diversity of this training dataset allow these
models to perform well on diverse tasks, including language syntax, semantics, and
context—even some requiring general knowledge.

# *** GPT-3 dataset details
Table 1.1 displays the dataset used for GPT-3. The proportions column in the table
sums up to 100% of the sampled data, adjusted for rounding errors. Although the
subsets in the Number of Tokens column total 499 billion, the model was trained on
only 300 billion tokens. The authors of the GPT-3 paper did not specify why the model
was not trained on all 499 billion tokens.
For context, consider the size of the CommonCrawl dataset, which alone consists of
410 billion tokens and requires about 570 GB of storage. In comparison, later iterations
of models like GPT-3, such as Meta’s LLaMA, have expanded their training
scope to include additional data sources like Arxiv research papers (92 GB) and
StackExchange’s code-related Q&As (78 GB).
The authors of the GPT-3 paper did not share the training dataset, but a comparable
dataset that is publicly available is Dolma: An Open Corpus of Three Trillion Tokens for
LLM Pretraining Research by Soldaini et al. 2024 (https://arxiv.org/abs/2402.00159).
However, the collection may contain copyrighted works, and the exact usage terms
may depend on the intended use case and country.

