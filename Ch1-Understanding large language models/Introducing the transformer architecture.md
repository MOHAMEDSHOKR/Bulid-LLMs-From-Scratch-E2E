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


The pretrained nature of these models makes them incredibly versatile for further
fine-tuning on downstream tasks, which is why they are also known as base or foundation
models. Pretraining LLMs requires access to significant resources and is very
expensive. For example, the GPT-3 pretraining cost is estimated to be $4.6 million in
terms of cloud computing credits (https://mng.bz/VxEW).

The good news is that many pretrained LLMs, available as open source models,
can be used as general-purpose tools to write, extract, and edit texts that were not
part of the training data. Also, LLMs can be fine-tuned on specific tasks with relatively
smaller datasets, reducing the computational resources needed and improving
performance.

We will implement the code for pretraining and use it to pretrain an LLM for educational
purposes. All computations are executable on consumer hardware. After implementing
the pretraining code, we will learn how to reuse openly available model weights
and load them into the architecture we will implement, allowing us to skip the expensive
pretraining stage when we fine-tune our LLM.

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

# 1.6 A closer look at the GPT architecture

GPT was originally introduced in the paper “Improving Language Understanding by
Generative Pre-Training” (https://mng.bz/x2qg) by Radford et al. from OpenAI.
GPT-3 is a scaled-up version of this model that has more parameters and was trained
on a larger dataset. In addition, the original model offered in ChatGPT was created by
fine-tuning GPT-3 on a large instruction dataset using a method from OpenAI’s
InstructGPT paper (https://arxiv.org/abs/2203.02155). As figure 1.6 shows, these
models are competent text completion models and can carry out other tasks such as
spelling correction, classification, or language translation. This is actually very remarkable
given that GPT models are pretrained on a relatively simple next-word prediction
task, as depicted in figure 1.7.

![image](https://github.com/user-attachments/assets/01d1bbcc-9715-43dd-807b-fc9c59639b49)

The next-word prediction task is a form of self-supervised learning, which is a form of
self-labeling. This means that we don’t need to collect labels for the training data
explicitly but can use the structure of the data itself: we can use the next word in a sentence
or document as the label that the model is supposed to predict. Since this nextword
prediction task allows us to create labels “on the fly,” it is possible to use massive
unlabeled text datasets to train LLMs.

Compared to the original transformer architecture we covered in section 1.4, the
general GPT architecture is relatively simple. Essentially, it’s just the decoder part
without the encoder (figure 1.8). Since decoder-style models like GPT generate text
by predicting text one word at a time, they are considered a type of autoregressive
model.

Autoregressive models incorporate their previous outputs as inputs for future predictions.
Consequently, in GPT, each new word is chosen based on the sequence
that precedes it, which improves the coherence of the resulting text.
Architectures such as GPT-3 are also significantly larger than the original transformer
model. For instance, the original transformer repeated the encoder and decoder blocks
six times. GPT-3 has 96 transformer layers and 175 billion parameters in total.

![image](https://github.com/user-attachments/assets/1e8e47da-5c24-4b65-b323-ab36fb3025c3)
Figure 1.8 The GPT architecture employs only the decoder portion of the original transformer. It is designed for
unidirectional, left-to-right processing, making it well suited for text generation and next-word prediction tasks to
generate text in an iterative fashion, one word at a time.

GPT-3 was introduced in 2020, which, by the standards of deep learning and large language
model development, is considered a long time ago. However, more recent architectures,
such as Meta’s Llama models, are still based on the same underlying concepts,
introducing only minor modifications. Hence, understanding GPT remains as relevant
as ever, so I focus on implementing the prominent architecture behind GPT while providing
pointers to specific tweaks employed by alternative LLMs.

Although the original transformer model, consisting of encoder and decoder blocks,
was explicitly designed for language translation, GPT models—despite their larger yet
simpler decoder-only architecture aimed at next-word prediction—are also capable of
performing translation tasks. This capability was initially unexpected to researchers, as
it emerged from a model primarily trained on a next-word prediction task, which is a
task that did not specifically target translation.

The ability to perform tasks that the model wasn’t explicitly trained to perform is
called an emergent behavior. This capability isn’t explicitly taught during training but
emerges as a natural consequence of the model’s exposure to vast quantities of multilingual
data in diverse contexts. The fact that GPT models can “learn” the translation
patterns between languages and perform translation tasks even though they weren’t
specifically trained for it demonstrates the benefits and capabilities of these largescale,
generative language models. We can perform diverse tasks without using diverse
models for each.

# 1.7 Building a large language model

Now that we’ve laid the groundwork for understanding LLMs, let’s code one from
scratch. We will take the fundamental idea behind GPT as a blueprint and tackle this
in three stages, as outlined in figure 1.9.
![image](https://github.com/user-attachments/assets/889ad3cc-31be-4c1a-a46f-0f7524d4d93f)
Figure 1.9 The three main stages of coding an LLM are implementing the LLM architecture and data preparation
process (stage 1), pretraining an LLM to create a foundation model (stage 2), and fine-tuning the foundation
model to become a personal assistant or text classifier (stage 3).

*** In stage 1, we will learn about the fundamental data preprocessing steps and code the
attention mechanism at the heart of every LLM. Next, 
*** in stage 2, we will learn how to code and pretrain a GPT-like LLM capable of generating new texts.
We will also go over the fundamentals of evaluating LLMs, which is essential for developing capable
NLP systems.
Pretraining an LLM from scratch is a significant endeavor, demanding thousands
to millions of dollars in computing costs for GPT-like models. Therefore, the focus of
stage 2 is on implementing training for educational purposes using a small dataset.
In addition, I also provide code examples for loading openly available model weights.

*** Finally, in stage 3, we will take a pretrained LLM and fine-tune it to follow instructions
such as answering queries or classifying texts—the most common tasks in many
real-world applications and research.
I hope you are looking forward to embarking on this exciting journey!


# Summary
 LLMs have transformed the field of natural language processing, which previously
mostly relied on explicit rule-based systems and simpler statistical methods.
The advent of LLMs introduced new deep learning-driven approaches
that led to advancements in understanding, generating, and translating human
language.
 Modern LLMs are trained in two main steps:
– First, they are pretrained on a large corpus of unlabeled text by using the
prediction of the next word in a sentence as a label.
– Then, they are fine-tuned on a smaller, labeled target dataset to follow
instructions or perform classification tasks.
 LLMs are based on the transformer architecture. The key idea of the transformer
architecture is an attention mechanism that gives the LLM selective
access to the whole input sequence when generating the output one word at
a time.
 The original transformer architecture consists of an encoder for parsing text
and a decoder for generating text.
 LLMs for generating text and following instructions, such as GPT-3 and
ChatGPT, only implement decoder modules, simplifying the architecture.
 Large datasets consisting of billions of words are essential for pretraining
LLMs.
 While the general pretraining task for GPT-like models is to predict the next
word in a sentence, these LLMs exhibit emergent properties, such as capabilities
to classify, translate, or summarize texts.
 Once an LLM is pretrained, the resulting foundation model can be fine-tuned
more efficiently for various downstream tasks.
 LLMs fine-tuned on custom datasets can outperform general LLMs on specific
tasks.

