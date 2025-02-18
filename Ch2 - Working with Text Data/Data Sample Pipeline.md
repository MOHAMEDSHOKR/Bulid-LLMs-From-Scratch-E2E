# Introduction to Data prepration & Sampling 

#### So far, we’ve covered the general structure of large language models (LLMs) and learned that they are pretrained on vast amounts of text.
#### Specifically, our focus was on "decoder-only" LLMs based on the transformer architecture, which underlies the models 
#### used in ChatGPT and other popular GPT-like LLMs.

## During the pretraining stage, LLMs process text one word at a time.
Training LLMs with millions to billions of parameters using a next-word prediction task
yields models with impressive capabilities.
These models can then be further finetuned to follow general instructions or perform specific target tasks.
But before we can implement and train LLMs, we need to prepare the training dataset, as illustrated in figure 2.1.

![image](https://github.com/user-attachments/assets/8dd0c282-7f73-4a76-87a7-d41360a36349)
Figure 2.1 The three main stages of coding an LLM. This chapter focuses on step 1 of stage 1: implementing the
data sample pipeline.

# What you will learn from this Chapter 
###  * You’ll learn how to prepare input text for training LLMs. 
###  * This involves splitting text into individual word and subword tokens, which can then be encoded into vector representations for the LLM. 
###  * You’ll also learn about advanced tokenization schemes like byte pair encoding, which is utilized in popular LLMs like GPT. 
###  * Lastly, we’ll implement a sampling and data-loading strategy to produce the input-output pairs necessaryfor training LLMs.
