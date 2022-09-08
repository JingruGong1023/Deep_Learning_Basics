## Introduction

Sequence models are the machine learning models that input or output sequences of data. Sequential data

includes text streams, audio clips, video clips, time-series data and etc.

In this chapter we will review some basics of Neural Networks , then dig deep for RNN

</br>

## Table of Contents

- [Some recap of Neural Networks](#Some_recap_of_Neural_Networks)
  - [Regularization](#Regularization)
  - [Dropout](#Dropout)
  - [Parameter Initialization](#Parameter Initialization)
- [Language Modeling](#Language_Modeling)
  - [n-gram Language Models](#n-gram Language Models)
- [RNNs](#RNNs)
  - [RNN Language Models](#RNN Language Models)

- [Exploding and Vanishing Gradients](#Exploding and Vanishing Gradients)
- [LSTMs](#LSTMs)
- [Bidirectional and multi-layer RNN](#Bidirectional and multi-layer RNN)







</br>

## Some recap of Neural Networks

**What is Neural Network?**

It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain.

It feeds inputs through different hidden layers and relies on weights and nonlinear functions to reach an output

<img src="../../image/Screen Shot 2022-08-07 at 10.23.15 PM.png" alt="Screen Shot 2022-08-07 at 10.23.15 PM" width = "50%" height = "50%" />

**General Methodology/ step to build a NN :**

1. Define the NN structure (# of input, # of hidden units, etc)

2. Initialize the model's parameter

3. Loop:

   a. Implement forward propagation -> compute activation for each layer and estimate y hat

   b. Compute loss

   c. Implement backward propagation to get the gradients

   d. update the parameters

   

### Regularization

How do we see regularization in traditional ML?

**Underfitting**: Model too simple -> high bias 

**Overfitting**: Model too complex -> high variance -> apply `Regularization`

**What is L2-regularization actually doing?**:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

**Traditional View**: Regularization works to prevent overfitting when we have a lot of features (or later a very powerful/deep model, etc.)

**For modern NN**: 

Regularization produces models that generalize well when we have a “big” model • We do not care that our models overfit on the training data, even though they are hugely overfit

<img src="../../image/Screen Shot 2022-08-07 at 10.33.36 PM.png" alt="Screen Shot 2022-08-07 at 10.33.36 PM" width = "30%" height = "30%" />

We don't want $\lambda$ to be too large so that the model is too simple and not work well, and we don't want it to be too small to not working on generalization

 

### Dropout

Preventing Feature Co-adaptation = Good Regularization Method! Use it widely!

- **Training time**: at each instance of evaluation (in online SGD-training), randomly set 50% of the inputs to each neuron to 0 
- **Test time**: halve the model weights (now twice as many)
- This prevents feature **co-adaptation**: A feature cannot only be useful in the presence of particular other features 
- In a single layer: A kind of middle-ground between Naïve Bayes (where all feature weights are set independently) and logistic regression models (where weights are set in the context of all others) 
- Can be thought of as a form of **model bagging** (i.e., like an ensemble model) 
- Nowadays usually thought of as strong, feature-dependent regularizer (Meaning, unlike uniformly regularization penalizes all features, Dropout penalizes features that uses less more. )



### Parameter Initialization

We want to **initialize weights** randomly instead of 0 as we did in logistic regression, because 0 will make the nodes in the hidden layer symmetric

You normally must initialize weights to **small random values**

Initialize hidden layer **biases** to 0 and output (or reconstruction) biases to optimal value if weights were 0 (e.g., mean target or inverse sigmoid of mean target)

Why do we want w to be close to 1? -> because when w is a little larger or smaller than x , with a DNN, y will explode or vanish

`Xaviers initialization` : $w^l = np.random.randn(n_h, n_x)* \sqrt{\frac{1}{dim \ of \ the \ previous \ layer}}$

`He initialization` : $w^l = np.random.randn(n_h, n_x)* \sqrt{\frac{2}{dim \ of \ the \ previous \ layer}}$  for ReLU

 

## Language Modeling

 `Language Model` is the task of predicting what word comes next
Given a sentence input, it will create a dictionary for the words (**tokenize**, similar as OHE), put a token as <EOS> to specify the end point of a sentence, punctuations can be added to the dictionary as well

More formally: given a sequence of words $x^{(1)}, x^{(2)}...x^{(t)}$, compute the probability distribution of the next word $x^{(t+1)}$:

​												$P(x^{(t+1)}| x^{(t)},...,x^{(1)})$

You can also think of a Language Model as a system that **assigns a probability to a piece of text**

For example, if we have some text $x^{(1)}, x^{(2)}...x^{(T)}$, then the probability of this text (according to the Language Model) is:

$P(x^{(1)}, x^{(2)}...x^{(T)}) = p(x^{(1)})* P(x^{(2)}|x^{(1)})*...*p(x^{(T)}|x^{(T-1)},...,x^{(1)})$

​								     $= \prod_{t = 1}^{T} P(x^{(t+1)}| x^{(t)},...,x^{(1)})$

But how do we learn a Language Model? -> n-gram!



### n-gram Language Models

**Definition**: An n-gram is a chunk of n consecutive words. 

- unigrams: “the”, “students”, “opened”, ”their” 
- bigrams: “the students”, “students opened”, “opened their” 
- trigrams: “the students opened”, “students opened their” 
- four-grams: “the students opened their”

**Idea**: Collect statistics about how **frequent** different n-grams are and use these to predict next word.

First we make a `Markov assumption`: $x^{(t+1)}$depends only on the preceding n-1 words

<img src="../../image/Screen Shot 2022-08-07 at 11.16.37 PM.png" alt="Screen Shot 2022-08-07 at 11.16.37 PM" height = "40%" width = "40%" />

so we throw away some early words, to only use the preceding n-1 words to predict 

**Question: How do we get these n-gram and (n-1)-gram probabilities? **

Answer: By counting them in some large corpus of text!

For example, suppose we are learning a 40gram Language Model

$P(w|students \ opened \ their) = \frac{count(student \ opended \ their \ w)}{count(student \ opended \ their)}$

For example, suppose in the corpus:

"Students opened their" occurred 1000 times, "Students opened their books" ocurred 400 times -> P(books|students opened their) = 0.4

#### Sparsity Problems with n-gram Language Models

**Sparsity Problem 1**: What if “students opened their w” never occurred in data? Then w has probability 0!

Partial Solution: Add small $\delta$ to the count for every w ∈ V. This is called smoothing.

**Sparsity Problem 2:** What if “students opened their” never occurred in data? Then we can’t calculate probability for any w!

Partial Solution : ust condition on “opened their” instead. This is called `backoff`; **Increasing n makes sparsity problems worse**. Typically, we can’t have n bigger than 5. In practice, we usually use **tri-gram**

#### Storage Problems with n-gram Language Models

Storage: Need to store count for all n-grams you saw in the corpus. 

Partial Solution : Increasing n or increasing corpus increases model size!

</br>

## RNNs

`Recurrent neural networks`, also known as `RNNs`, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states.

It predicts sequential data using a temporally connected system that captures both new inputs and precious outputs using hidden states

**Core Idea: Apply the same weights repeatedly**

**Why not just use standard Network?**

- Input and output can be different lengths in different examples
- Doesn't share features learned across different positions of text

<img src="../../image/Screen Shot 2022-08-08 at 10.34.24 AM.png" alt="Screen Shot 2022-08-08 at 10.34.24 AM" height = "50%" width = "50%" />

Here, $h^{(t)}$ means the hidden state , and e means the word embedding vector

So, with applying the **same weights** to hidden state and word embeddings at each time step, we take the information from **previous hidden state** and **current word embeddings** add with a learned bias, apply it to an non-linear function -> $h^{(t)}$

**RNN Advantages:** 

	- Can process any length input
	- Computation for step t can (in theory) use information from many steps back 
	- Model size doesn’t increase for longer input context
	- Same weights applied on every timestep, so there is symmetry in how inputs are processed.

**RNN Disadvantages: **

	- Recurrent computation is slow
	- In practice, difficult to access information from many steps back 



### RNN Language Models

- Get a big corpus of text which is a sequence of words $x^{(1)}, x^{(2)},...$
- Feed into RNN LM ; Compute the output of distribution y hat for every step t -> **predict probability dist of every word, given word so far**
- Loss Function on step t is **corss entropy** between predicted probability distribution, and the true next word for each step t
- Average this to get the overall loss for entire training set

<img src="../image/Screen Shot 2022-09-06 at 3.27.10 PM.png" alt="Screen Shot 2022-09-06 at 3.27.10 PM" height = "50%" width = "50%" />

**Training tips**

Note that, computing loss and gradients across entire corpus $x^{(1)}, x^{(2)},...$ can be really expensive

Instead, we should compute loss $J(\theta)$  for a batch of sentence, and then compute gradients and update weights, repeat.

#### backpropagation

<img src="../image/Screen Shot 2022-09-06 at 3.35.30 PM.png" alt="Screen Shot 2022-09-06 at 3.35.30 PM" height = "20%" width = "20%" />

#### Generating text with a RNN Language Model

Just like a `n-gram Language Model`, you can use an RNN Language Model to generate text by **repeated sampling**. Sampled output becomes next step’s input.

<img src="../image/Screen Shot 2022-09-06 at 3.42.44 PM.png" alt="Screen Shot 2022-09-06 at 3.42.44 PM" height = "50%" width = "50%"/>

When generating text we choose only **one** of the words ourselves given the probabilities and feed that back into the network. This is called **sampling**.
 To begin character-level sampling:

- Input a "dummy" vector of zeros as a default input

- Run one step of forward propagation to get a⟨1⟩ (your first character) and y_hat ⟨1⟩ (probability

distribution for the following character)

- When sampling, avoid generating the same result each time given the starting letter by

using *np.random.choice*

#### Evaluating Language Models

The standard **Evaluation metric** for LM is **Perplexity**

<img src="../image/Screen Shot 2022-09-06 at 3.50.13 PM.png" alt="Screen Shot 2022-09-06 at 3.50.13 PM" height = "50%" width = "50%" />

`Perplexity` is the inverse of the probability of t+1 th word given the previous t words in the sentence.

Since the nature of inverse, we expect the Perplexity **the lower the better**

$Perplexity = exp(J(\theta))$

#### Summary

`Language Model`:

A system that predicts the next word 

`Recurrent Neural Network`:

A family of neural networks that: 

- Take sequential input of any length 
- Apply the same weights on each step 
- Can optionally produce output on each step

**Recurrent Neural Network ≠ Language Model**

 We’ve shown that RNNs are a great way to build a LM , **But RNNs are useful for much more!**

Such as sentiment classification, question answering

***However , RNN is not perfect***

</br>

## Exploding and Vanishing Gradients

**Vanishing gradient problem**: When these are small, the gradient signal gets smaller and smaller as it backpropagates further -> because of chain rule

**Gradient signal from far away is lost** because it’s much smaller than gradient signal from close-by. So, model weights are updated only with respect to near effects, not long-term effects. So we can conclude basic RNN is not good at long-term distance model dependency

For example, when we have a sentence input “The cats, which already ate ... , were at home”, the “Cats” at the very beginning of the sentence may has a hard time affect “were”, which at the end of the sentence. So that, when we do backpropagation, the calculation of the end time step, may not be able to affect the gradients in the beginning of the time step.



**Solution**: `LSTM`

**Exploding Gradient Problem**: If the gradient becomes too big, then the SGD update step becomes too big

<img src="../image/Screen Shot 2022-09-07 at 10.06.39 PM.png" alt="Screen Shot 2022-09-07 at 10.06.39 PM" height = "30%" width = "30%" />

This can cause bad updates: we take too large a step and reach a weird and bad parameter configuration (with large loss)

 - In the worst case, this will result in Inf or NaN in your network 

**Solution** : `Gradient Clipping`

if the norm of the gradient is greater than some threshold, scale it down before applying SGD update

- take a step in the same direction, but a **smaller step**



## LSTM (Long short-term Memory RNNs)

**On step t，there is a hidden state and a cell state **

	- Both are vectors length n
	- The cell stores long-term information
	- The LSTM can read, erase, and write information from the cell
	- The cell becomes conceptually rather like RAM in a computer

The selection of which information is erased/written/read is controlled by three corresponding gates：

	- The gates are also vectors of length n
	- On each timestep, each element of the gates can be open (1), closed (0), or somewhere in-between -> so we will use Sigmoid
	- The gates are dynamic: their value is computed based on the current context 

#### Computation Steps at time step t

 <img src="../image/Screen Shot 2022-09-07 at 10.44.19 PM-2605130.png" alt="Screen Shot 2022-09-07 at 10.44.19 PM" height = "80%" width = "80%"/>

Notice that, none of the four new gates (including Forget gate, input gate, output gate and new cell content) is calculated dependently, so in practice, we can calculate them in parallel.

**If we put the calculation in visualization**

<img src="../image/Screen Shot 2022-09-07 at 10.51.58 PM.png" alt="Screen Shot 2022-09-07 at 10.51.58 PM" height = "80%" width = "80%" />

**LSTM doesn’t guarantee that there is no vanishing/exploding gradient**, but it does provide an easier way for the model to learn long-distance dependencies



### Gated Recurrent Unit (GRU)

Another common solution to Vanishing gradients, is GRU. 

To solve the vanishing gradient problem of a standard RNN, GRU uses **update gate** and **reset gate,** deciding which and what information should be passed to the output. With these two trained parameters, we can save information for a long term, like from the starting time step to the end time step. This is called **the Gated Hidden State,** same as the output of activation function

1. **Update gate**

update gate would allow us to control how much of the new state is just a copy of the old state.
 We start with calculating the update gate. we use sigmoid function to keep the value close to 0 or 1
memory cell may contain information like plural or singular, and update gate has the ability to know when to update/ change the value of memory cell. For example, in the sentence “The cat, which ate ..., was full”, the memory cell change to 1 at “cat” indicating singular, till “was”, and the update gate will know it’s used, and we no longer need this information, and will forget it.

2. **Calculate the candidate to replace memory cell**

3. **Calculate memory cell value**

   <img src="../image/Screen Shot 2022-09-07 at 11.01.11 PM.png" alt="Screen Shot 2022-09-07 at 11.01.11 PM" height = "30%" width = "30%"/>

so in this example, we set the Γ =1 at cat, so we update c there, and keep Γ = 0 all the way till “was” , basically saying “don’t update, remember the previous c, so that in the equation , which uses the previous value of memory cell.

### GRU VS LSTM

**Rule of thumb:** LSTM is a good default choice (especially if your data has particularly long dependencies, or you have lots of training data); **Switch to GRUs for speed and fewer parameters.**



## Bidirectional and multi-layer RNN

### Bidirectional RNN

No matter for LSTM, GRU, or basic RNN, they all are forward process, but sometimes it’s not enough to predict the following words only based on the previous words. For example, with only knowing “Teddy”, we are not sure if it’s “Teddy bear” or it’s “Teddy Roosevelt”

**BRNN allows us to take both previous and future output as input to make prediction**

<img src="../image/Screen Shot 2022-09-07 at 11.10.54 PM.png" alt="Screen Shot 2022-09-07 at 11.10.54 PM" height = "50%" width = "50%" />



<img src="../image/Screen Shot 2022-09-07 at 11.11.50 PM.png" alt="Screen Shot 2022-09-07 at 11.11.50 PM" height = "50%" width = "50%"  />

**Note:** bidirectional RNNs are only applicable if you have access to the **entire input sequence **

 - They are not applicable to Language Modeling, because in LM you only have left context available.

 - For example, `BERT (Bidirectional Encoder Representations from Transformers)` is a powerful pretrained contextual representation system built on bidirectionality.

   
