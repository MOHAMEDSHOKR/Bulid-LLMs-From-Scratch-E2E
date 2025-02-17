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


LLMs utilize an architecture called the transformer, which allows them to pay selective
attention to different parts of the input when making predictions, making them
especially adept at handling the nuances and complexities of human language.
Since LLMs are capable of generating text, LLMs are also often referred to as a form
of generative artificial intelligence, often abbreviated as generative AI or GenAI. As illustrated
in figure 1.1, AI encompasses the broader field of creating machines that can
perform tasks requiring human-like intelligence, including understanding language,
recognizing patterns, and making decisions, and includes subfields like machine
learning and deep learning.


![image](https://github.com/user-attachments/assets/b4bfb91a-6d31-4384-ad4e-2aef42dd5a16)
Figure 1.1 As this hierarchical depiction of the relationship between the different fields suggests, LLMs
represent a specific application of deep learning techniques, using their ability to process and generate humanlike
text. Deep learning is a specialized branch of machine learning that focuses on using multilayer neural
networks. Machine learning and deep learning are fields aimed at implementing algorithms that enable computers
to learn from data and perform tasks that typically require human intelligence.

The algorithms used to implement AI are the focus of the field of machine learning.
Specifically, machine learning involves the development of algorithms that can learn
from and make predictions or decisions based on data without being explicitly programmed.
To illustrate this, imagine a spam filter as a practical application of
machine learning. Instead of manually writing rules to identify spam emails, a
machine learning algorithm is fed examples of emails labeled as spam and legitimate
emails. By minimizing the error in its predictions on a training dataset, the model
then learns to recognize patterns and characteristics indicative of spam, enabling it to
classify new emails as either spam or not spam.

As illustrated in figure 1.1, deep learning is a subset of machine learning that focuses
on utilizing neural networks with three or more layers (also called deep neural networks)
to model complex patterns and abstractions in data. In contrast to deep learning,
traditional machine learning requires manual feature extraction. This means that
human experts need to identify and select the most relevant features for the model.

While the field of AI is now dominated by machine learning and deep learning, it
also includes other approaches—for example, using rule-based systems, genetic algorithms,
expert systems, fuzzy logic, or symbolic reasoning.
Returning to the spam classification example, in traditional machine learning,
human experts might manually extract features from email text such as the frequency
of certain trigger words (for example, “prize,” “win,” “free”), the number of
exclamation marks, use of all uppercase words, or the presence of suspicious links.
This dataset, created based on these expert-defined features, would then be used to
train the model. In contrast to traditional machine learning, deep learning does not
require manual feature extraction. This means that human experts do not need to
identify and select the most relevant features for a deep learning model. (However,
both traditional machine learning and deep learning for spam classification still
require the collection of labels, such as spam or non-spam, which need to be gathered
either by an expert or users.)
Let’s look at some of the problems LLMs can solve today, the challenges that LLMs
address, and the general LLM architecture we will implement later.

# 1.2 Applications of LLMs

Owing to their advanced capabilities to parse and understand unstructured text data,
LLMs have a broad range of applications across various domains. Today, LLMs are
employed for machine translation, generation of novel texts (see figure 1.2), sentiment
analysis, text summarization, and many other tasks. LLMs have recently been
used for content creation, such as writing fiction, articles, and even computer code.
LLMs can also power sophisticated chatbots and virtual assistants, such as OpenAI’s
ChatGPT or Google’s Gemini (formerly called Bard), which can answer user queries
and augment traditional search engines such as Google Search or Microsoft Bing.

Moreover, LLMs may be used for effective knowledge retrieval from vast volumes
of text in specialized areas such as medicine or law. This includes sifting through documents,
summarizing lengthy passages, and answering technical questions.
In short, LLMs are invaluable for automating almost any task that involves parsing
and generating text. Their applications are virtually endless, and as we continue to
innovate and explore new ways to use these models, it’s clear that LLMs have the
potential to redefine our relationship with technology, making it more conversational,
intuitive, and accessible.

We will focus on understanding how LLMs work from the ground up, coding an
LLM that can generate texts. You will also learn about techniques that allow LLMs to
carry out queries, ranging from answering questions to summarizing text, translating
text into different languages, and more. In other words, you will learn how complex
LLM assistants such as ChatGPT work by building one step by step.

# 1.3 Stages of building and using LLMs

Why should we build our own LLMs? Coding an LLM from the ground up is an excellent
exercise to understand its mechanics and limitations. Also, it equips us with the
required knowledge for pretraining or fine-tuning existing open source LLM architectures
to our own domain-specific datasets or tasks.

*** NOTE Most LLMs today are implemented using the PyTorch deep learning
library, which is what we will use.


Research has shown that when it comes to modeling performance, custom-built
LLMs—those tailored for specific tasks or domains—can outperform general-purpose
LLMs, such as those provided by ChatGPT, which are designed for a wide array of
applications. Examples of these include BloombergGPT (specialized for finance) and
LLMs tailored for medical question answering


Using custom-built LLMs offers several advantages, particularly regarding data privacy.
For instance, companies may prefer not to share sensitive data with third-party
LLM providers like OpenAI due to confidentiality concerns. Additionally, developing
smaller custom LLMs enables deployment directly on customer devices, such as laptops
and smartphones, which is something companies like Apple are currently exploring.


This local implementation can significantly decrease latency and reduce server-related
costs. Furthermore, custom LLMs grant developers complete autonomy, allowing
them to control updates and modifications to the model as needed.
The general process of creating an LLM includes pretraining and fine-tuning. The
“pre” in “pretraining” refers to the initial phase where a model like an LLM is trained
on a large, diverse dataset to develop a broad understanding of language. This pretrained
model then serves as a foundational resource that can be further refined
through fine-tuning, a process where the model is specifically trained on a narrower
dataset that is more specific to particular tasks or domains. This two-stage training
approach consisting of pretraining and fine-tuning is depicted in figure 1.3.

![image](https://github.com/user-attachments/assets/668c0660-f274-442a-bdd1-75173ff2762c)
Figure 1.3 Pretraining an LLM involves next-word prediction on large text datasets. A pretrained LLM
can then be fine-tuned using a smaller labeled dataset.

*** The first step in creating an LLM is to train it on a large corpus of text data, sometimes
referred to as raw text. Here, “raw” refers to the fact that this data is just regular text
without any labeling information. (Filtering may be applied, such as removing formatting
characters or documents in unknown languages.)

*** This first training stage of an LLM is also known as pretraining, creating an initial pretrained
LLM, often called a base or foundation model. A typical example of such a model
is the GPT-3 model (the precursor of the original model offered in ChatGPT). This
model is capable of text completion—that is, finishing a half-written sentence provided
by a user. It also has limited few-shot capabilities, which means it can learn to
perform new tasks based on only a few examples instead of needing extensive training
data.

*** After obtaining a pretrained LLM from training on large text datasets, where the
LLM is trained to predict the next word in the text, we can further train the LLM on
labeled data, also known as fine-tuning.

*** The two most popular categories of fine-tuning LLMs are instruction fine-tuning and
classification fine-tuning. In instruction fine-tuning, the labeled dataset consists of
instruction and answer pairs, such as a query to translate a text accompanied by the
correctly translated text. In classification fine-tuning, the labeled dataset consists of
texts and associated class labels—for example, emails associated with “spam” and “not
spam” labels.



