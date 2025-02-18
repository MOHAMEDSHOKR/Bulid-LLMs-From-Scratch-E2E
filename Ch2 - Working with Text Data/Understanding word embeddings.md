# 2.1 Understanding word embeddings

Deep neural network models, including LLMs, cannot process raw text directly. 
Since text is categorical, it isn’t compatible with the mathematical operations used to implement
and train neural networks. Therefore, we need a way to represent words as "continuous-valued vectors".

### The concept of converting data into a vector format is often referred to as " embedding ".
Using a specific neural network layer or another pretrained neural network model, we
can embed different data types—for example, video, audio, and text, as illustrated in figure 2.2. 
However, it’s important to note that different data formats require distinct embedding models.
For example, an embedding model designed for text would not be suitable for embedding audio or video data.
