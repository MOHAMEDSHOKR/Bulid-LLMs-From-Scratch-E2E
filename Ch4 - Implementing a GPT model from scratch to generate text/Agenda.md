# Implementing a GPT model from scratch to generate text

## This chapter covers

   1- Coding a GPT-like large language model (LLM) that can be trained to generate human-like text.
   
   2-  Normalizing layer activations to stabilize neural network training.
   
   3- Adding shortcut connections in deep neural networks.
   
   4- Implementing transformer blocks to create GPT models of various sizes.
   
   5- Computing the number of parameters and storage requirements of GPT models.



You’ve already learned and coded the multi-head attention mechanism, one of the core components of LLMs. 

Now, we will code the other building blocks of an LLM and assemble them into a GPT-like model that we will train in the next chapter to generate human-like text.
The LLM architecture referenced in figure 4.1, consists of several building blocks.
We will begin with a top-down view of the model architecture before covering the individual components in more detail.

-------------------------------------------------------------------------------------------------------------------------------

![image](https://github.com/user-attachments/assets/374930ed-889e-40c4-886f-5c98ab124cbd)
Figure 4.1 The three main stages of coding an LLM. This chapter focuses on step 3 of stage 1: implementing the
LLM architecture.
