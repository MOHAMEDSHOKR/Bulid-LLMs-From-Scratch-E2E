# This chapter covers
 ### * The reasons for using attention mechanisms inneural networks.
### * A basic self-attention framework, progressing to an enhanced self-attention mechanism.
### * A causal attention module that allows LLMs to generate one token at a time.
### * Masking randomly selected attention weights with dropout to reduce overfitting.
### * Stacking multiple causal attention modules into a multi-head attention module.

# Introduction 
At this point, you know how to prepare the input text for training LLMs by splitting
text into individual word and subword tokens, which can be encoded into vector representations,
embeddings, for the LLM.
Now, we will look at an integral part of the LLM architecture itself, attention
mechanisms, as illustrated in figure 3.1. We will largely look at attention mechanisms
in isolation and focus on them at a mechanistic level. Then we will code the remaining parts of the LLM 
surrounding the self-attention mechanism to see it in action and to create a model to generate text.
We will implement four different variants of attention mechanisms, as illustrated in
figure 3.2. These different attention variants build on each other, and the goal is to arrive 
at a compact and efficient implementation of multi-head attention that we can
then plug into the LLM architecture we will code in the next chapter.

![image](https://github.com/user-attachments/assets/997b696f-8cb0-42ac-ad6d-1195583320ef)
Figure 3.1 The three main stages of coding an LLM. This chapter focuses on step 2 of stage 1: implementing
attention mechanisms, which are an integral part of the LLM architecture.

---------------------------------------------------------------------------------------------------

![image](https://github.com/user-attachments/assets/3b219c86-2d35-4a55-a6e7-6bb191517eee)
Figure 3.2 The figure depicts different attention mechanisms we will code in this chapter, starting
with a simplified version of self-attention before adding the trainable weights. The causal attention
mechanism adds a mask to self-attention that allows the LLM to generate one word at a time. Finally,
multi-head attention organizes the attention mechanism into multiple heads, allowing the model to
capture various aspects of the input data in parallel.
