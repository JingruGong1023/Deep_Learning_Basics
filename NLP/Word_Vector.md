## Introduction

The first and arguably most important common denominator across all NLP tasks is how we represent words as input to any of our models. Much of the earlier NLP work that we will not cover treats words as atomic symbols. To perform well on most NLP tasks we first need to have some notion of similarity and difference between words. With word vectors, we can quite easily encode this ability in the vectors themselves (using distance measures such as Jaccard, Cosine, Euclidean, etc).

## Table of Contents

-[1. Basic Word Vectors](#Word-Vectors)

-[2. SVD Based Methods](#SVD-Based-Methods)

​	-[2.1 Word-Document Matrix](#Word-Document Matrix)





<br>

## Word-Vectors

There are an estimated 13 million tokens for the English language but are they all completely unrelated? Feline to cat, hotel to motel? I think not. Thus, we want to encode word tokens each into some vector that represents a point in some sort of "word" space. This is paramount for a number of reasons but the most intuitive reason is that perhaps there actually exists some N-dimensional space (such that N  13 million) that is sufficient to encode all semantics of our language. Each dimension would encode some meaning that we transfer using speech.

The first and most simple word vector / word representation would be **One-Hot Vector**. Represent every word as an $R^|V|x1$. vector with all 0s and one 1 at the index of that word in the sorted english language. And |V| is the size of our vocabulary. 

We represent each word as a completely independent entity. As we previously discussed, this word representation does not give us directly any notion of similarity

## SVD-Based-Methods

For this class of methods to find word embeddings (otherwise known as word vectors), we first loop over a massive dataset and accumulate word co-occurrence counts in some form of a matrix X, and then perform Singular Value Decomposition on X to get a USV^T^ decomposition. We then use the rows of U as the word embeddings for all words in our dictionary. Let us discuss a few choices of X.




