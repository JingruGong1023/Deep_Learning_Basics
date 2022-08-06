## Table-of-Contents

- [1. Math Details](#Math-Details)
  - [1.1 DL Classification Task: Named Entity Regonition](#DL-Classification-Task-:-Named-Entity-Regonition)





## Math Details

**This section will go through the NN math details through examples**

### DL Classification Task Named Entity Regonition

Task: **find** and **classify** names in text, for example:

Last night , Paris Hilton wowed in a sequin gown in April 1989. ->  Here, **Paris** and **Hilton** will be classify as ***PERSON*** and **April** and **1989** will be classify as ***DATE***

**Possible uses:**

- Tracking mentions of particular entities in documents 
- For question answering, answers are usually named entities 
- Relating sentiment analysis to the entity under discussion

#### Simple NER: Window classification using binary logistic classifier

**Idea**:

**Classify** each word in its `context window` of neighboring words.

Train `logistic classifier` on **hand-labeled** data to classify center word **{yes/no}** for each class based on a concatenation of word vectors in a window (In reality, we should use Softmax instead of Logistic, but here just to keep it simple)

- To classify all words: run classifier for **each class** on the vector centered on **each word** in the sentence

For example, to classify ***Paris*** in the sentence below, as +/- location, with **window length = 2**

$X_{window} = [x_{museums} \ x_{in} \ x_{Paris} \ x_{are} \ x_{amazing}]^T$ -> resultsing vector  $X_{window} = x \in \ R^{5d}$  -> a column vector!

#### Neural Classification

[check here for math details of how to calculate gradients](https://web.stanford.edu/class/cs224n/slides/cs224n-2022-lecture03-neuralnets.pdf)

[check here for Neural Network Basics](https://github.com/JingruGong1023/Deep_Learning/blob/main/Neural%20Network%20Basics/Neural%20Networks%20and%20Deep%20Learning%20Notes.pdf)

















