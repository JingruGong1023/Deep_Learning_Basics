## Introduction

Parse trees in NLP, analogous to those in compilers, are used to analyze the syntactic structure of sentences. There are two main types of structures used - **constituency structures** and **dependency structures**.

`Constituency Grammar` uses phrase structure grammar to organize words into **nested constituents**. 

`Dependency structure` of sentences shows which words depend on (modify or are arguments of) which other words. These binary asymmetric relations between the words are called dependencies and are depicted as arrows going from the head (or governor, superior, regent) to the dependent (or modifier, inferior, subordinate). Usually these dependencies form a tree structure. They are often typed with the name of grammatical relations (subject, prepositional object, apposition, etc.).  Sometimes a fake ROOT node is added as the head to the whole tree so that every word is a dependent of exactly one node

## Table of Contents

- [Syntactic Structure: Consistency and Dependency](#Syntactic-Structure-Consistency-and-Dependency )
- [Dependency Grammar and Treebanks](#Dependency-Grammar-and-Treebanks)
  - [Dependency Grammar and Dependency Structure](#Dependency-Grammar-and-Dependency-Structure)
  - [Treebanks](#Treebanks)
- [Dependency Parsing](#Dependency-Parsing)
  - [Transition-based dependency parsing](#Transition-based-dependency-parsing)
- [Neural dependency parsing](#Neural-dependency-parsing)
  - [Graph-based dependency parsers](#Graph-based-dependency-parsers)



## Syntactic Structure: Consistency and Dependency 

Two views of linguistic structure: 

Phrase structures+Dependency Structure

#### 1.Constituency = phrase structure grammar = context-free grammars (CFGs)

Phrase structure organizes words into nested constituents

Starting with unit : words -> the, cat, cuddly, by ,door

words combine into phrases -> the cuddly cat, by the door

Phrases can combine into bigger phrases -> the cuddly cat by the door

For example:

Lexicon: N(**Noun**): dog, cat ; Det(**determiner**): a, the ; P(**preposition**): in ,on, by ; V(**verb**): talk, walked

Grammar: 

**NP(noun phrase)** -> Det+ (adj)+N+(PP) -> () means optional : examples: the large cat in a crate

**PP(Prepositional Phrase)** -> P+NP : in a crate; on the table

**VP(verb phrase)** -> V+PP : walked by the street

#### 2. Dependency Structure

Dependency structure shows which words **depend on** (modify, attach to, or are arguments of) which other words.

For example, ***Look in the large crate in the kitchen by the door***

large is the dependent of crate, in the crate is the dependent of look

**Why do we need sentence structure?**

Humans communicate complex ideas by composing words together into bigger units to convey complex meanings 

Listeners need to work out what **modifies [attaches to]** what 

A model needs to understand sentence structure in order to be able to interpret language correctly

Though we have different types of **ambiguity** in human languages

- Prepositional phrase attachment ambiguity
  - *"Scientists count whales from space"*
- Coordination scope ambiguity
  - *"Shuttle veteran and longtime NASA executive Fred Gregory appointed to board"*
- Adjectival/Adverbial Modifier Ambiguity
  - *"Students get first hand job experience"*
- Verb Phrase (VP) attachment ambiguity
  - *Multilated body washes up on beach to be used for Olympics beach volleyball*

</br>

## Dependency Grammar and Treebanks

### Dependency Grammar and Dependency Structure

Dependency syntax postulates that syntactic structure consists of relations between lexical items, normally binary asymmetric relations (“arrows”) called **dependencies**

<img src="../../image/Screen Shot 2022-08-07 at 1.55.38 PM.png" alt="Screen Shot 2022-08-07 at 1.55.38 PM" style="zoom:15%;" />

The arrows are commonly typed with the name of grammatical relations (subject, prepositional object, apposition, etc.)

*Some of the Universal Dependency relations ([de Marneffe et al., 2014](http://www.lrec-conf.org/proceedings/lrec2014/pdf/1062_Paper.pdf))*

<img src="../../image/Screen Shot 2022-08-07 at 1.55.53 PM.png" alt="Screen Shot 2022-08-07 at 1.55.53 PM" style="zoom:25%;" />

We can always show this structure in a sentence:

<img src="../../image/Screen Shot 2022-08-07 at 1.57.54 PM.png" alt="Screen Shot 2022-08-07 at 1.57.54 PM" style="zoom:20%;" />

Some people draw the arrows one way; some the other way! 

- Tesnière had them point from head to dependent – we follow that convention 
- We usually add a fake ROOT so every word is a dependent of precisely 1 other node

### Treebanks

A treebank is a collection of syntactically annotated sentences in which the annotation has been manually checked so that the treebank can serve as a training corpus for natural language parsers, as a repository for linguistic research, or as an evaluation corpus for NLP systems.

**Semantic Treebanks**

These Treebanks use a formal representation of sentence’s semantic structure. They vary in the depth of their semantic representation. Robot Commands Treebank, Geoquery, Groningen Meaning Bank, RoboCup Corpus are some of the examples of Semantic Treebanks.

**Syntactic Treebanks**

Opposite to the semantic Treebanks, inputs to the Syntactic Treebank systems are expressions of the formal language obtained from the conversion of parsed Treebank data. The outputs of such systems are predicate logic based meaning representation. Various syntactic Treebanks in different languages have been created so far. For example, **Penn Arabic Treebank, Columbia Arabic Treebank** are syntactic Treebanks created in Arabia language. **Sininca** syntactic Treebank created in Chinese language. **Lucy, Susane** and **BLLIP WSJ** syntactic corpus created in English language.

**Advantages**

- Reusability of the labor 
  - Many parsers, part-of-speech taggers, etc. can be built on it
- Valuable resource for linguistics 
  - Broad coverage, not just a few intuitions 
- Frequencies and distributional information 
- A way to evaluate NLP systems

While we create Treebanks, we follow the conditions:

1. Bilexical affinities

    The dependency [discussion->issues] is plausible 

2. Dependency distance 

   Most dependencies are between nearby words 

3. Intervening material 

   Dependencies rarely span intervening verbs or punctuation 

4. Valency of heads 

   How many dependents on which side are usual for a head?

</br>

## Dependency Parsing

A sentence is **parsed** by choosing for each word what other word (including ROOT) it is a dependent of

**Dependency parsing** takes in a sentence and produces a set of directed, labeled arcs from **head** to **dependents**. But what are heads and dependents? Intuitively, the dependency parser wants to identify the key concept in a sentence.

For example: ***I love you***

Here, the key concept is to **describe the action or state of ‘love’:** somebody loves something or somebody else. The word ‘love’ is a **head**, while ‘I’ and ‘you’ are **dependents** that decorate or, in other words, specify the action of love. 

#### Methods of Dependency Parsing

1. Dynamic programming 

   Eisner (1996) gives a clever algorithm with complexity O(n3), by producing parse items with heads at the ends rather than in the middle 

2. Graph algorithms 

   You create a Minimum Spanning Tree for a sentence 

   McDonald et al.’s (2005) MSTParser scores dependencies independently using an ML classifier (he uses MIRA, for online learning, but it can be something else) Neural graph-based parser: Dozat and Manning (2017) et seq. – very successful! 

3. Constraint Satisfaction 

   Edges are eliminated that don’t satisfy hard constraints. Karlsson (1990), etc. 

4. “Transition-based parsing” or “deterministic dependency parsing” 

   Greedy choice of attachments guided by good machine learning classifiers 

   E.g., MaltParser (Nivre et al. 2008). Has proven highly effective.

**Problems with traditional parsers**

1. From a statistical perspective, these parsers suffer from the use of millions of **mainly poorly estimated feature weights**.

2. Almost all existing parsers rely on a **manually designed** s et of feature templates, which require a lot of expertise and are usually **incomplete**.

3. The use of many feature templates causes a less studied problem: in modern dependency parsers, most of the **runtime** is consumed not by the core parsing algorithm but in the feature extraction step.

### Transition-based dependency parsing

A simple form of greedy discriminative dependency parser

The parser does a sequence of bottom-up actions 

- Roughly like “shift” or “reduce” in a shift-reduce parser, but the “reduce” actions are specialized to create dependencies with head on left or right

Transition-based dependency parsing aims to predict a transition sequence from an initial configuration to some terminal configuration. The configuration looks like this:

$Configuration c = (s,b,A)$

- s(stack): records what’s undecided so far
- b(buffer): records what's not looked at all by our algorithm
- A(set of arcs): records the predictions, namely how arrows jump between the words in a sentence starting from ROOT

<img src="../../image/Screen Shot 2022-08-07 at 2.58.29 PM.png" alt="Screen Shot 2022-08-07 at 2.58.29 PM" style="zoom:25%;" />

The above chart shows how an arc-standard system deals with the sentence: "He has good control."

- **Shift**: move words sequentially into stack for consideration

  We start with the initial configuration: 

  [an **empty stack** with ROOT only, a full buffer with the whole sentence, and the empty set A]

  Starting from ‘He’, we sequentially move words into the stack and then think if we can come up with a reasonable dependency arc. Remember that: words in the stack are the **only** words we look at. If we could NOT come up with a dependency arc, we **shift** one word from the buffer to the stack.

- **Left-arc**: the right most two words in the stack has a dependency arc

  When we have [ROOT He has] in the stack, for the right-most words: ‘He’ and ‘has’, we know that ‘He’ is a word that specifies ‘has’, which is the key concept of this example sentence. It means that ‘has’ is the **head** while ‘He’ is the **dependent**. 

  Then we can come up with the next transition: left-arc, for that to point from the head ‘has’ to the dependent ‘He’, we need a left arc. Then we remove the dependent from the stack. For CS folks, it’s usually uncomfortable at first because the stack data structure usually performs in a FIFO (first-in-first-out) fashion, but here it’s NOT. We remove the dependent no matter it’s the right-most or second-right-most word.

- **Right-arc**: the left most two words in the stack has a dependency arc

- **Stop when there's only ROOT in the stack again**

we are left with the question: **How to choose the next action? Ie, how to choose between left-arc and right-arc** -> [MaltParser](https://www.maltparser.org/intro.html) -> Each action is predicted by a discriminative classifier (e.g., softmax classifier) over each legal move

but an even better solution -> **Neural Network Based Parser**!

## Neural dependency parsing

**Neural Approach**: Learn a **dense** and **compact** feature representation

The neural network here serves as the brain in the **‘come up with a dependency arc based on right-most two words in stack’** process. It takes in the dense representation vector of the words and predicts the transition in the softmax fashion.

the Network uses concatenated embeddings for **[words/POS tags/arc labels]** as the input.

The training set is generated using the**“shortest stack” **oracle which always prefers **LEFT-ARC** over **SHIFT**. The detail is not important here, but we should know that for each possible configuration there’s a ground truth transition. And we train our neural network parser in a **supervised** fashion.

<img src="../../image/Screen Shot 2022-08-07 at 4.14.20 PM.png" alt="Screen Shot 2022-08-07 at 4.14.20 PM" style="zoom:25%;" />

Steps in training:

- Cube activation function
- Choose top 3 words both in the stack and buffer
- Mini-batched AdaGrad with dropout
- Choose the parameters with the highest unlabeled attachment score

Given a configuration, we first extract representations for words/tags/labels chosen, then use it as the input of our neural network to predict the next transition, and finally generate a new configuration based on the prediction.

**Learn with Deep Learning classifiers - non-linear**

Traditional ML classifiers (including `Naïve Bayes, SVMs, logistic regression and softmax classifier`) are not very powerful classifiers: they only give **linear decision boundaries**

- `Neural networks` can learn much more complex functions with nonlinear decision boundaries

#### Simple feed-forward neural network multi-class classifier

we use non-linear activation functions(ReLU, Sigmoid) to deal with the data and get it ready for the linear classifier(softmax) in the output layer

<img src="../../image/Screen Shot 2022-08-07 at 4.17.14 PM.png" alt="Screen Shot 2022-08-07 at 4.17.14 PM" style="zoom:40%;" />

Neural networks can accurately determine the **structure of sentences**, supporting interpretation

The dense representations (and non-linear classifier) let it outperform other greedy parsers in both **accuracy** and **speed**



### Graph-based dependency parsers

Compute a score for every possible dependency for each word 

- Doing this well requires good “contextual” representations of each word token
- And repeat the same process for each other word
- Robust, but **slow**

<img src="../../image/Screen Shot 2022-08-07 at 4.21.27 PM.png" alt="Screen Shot 2022-08-07 at 4.21.27 PM" style="zoom:25%;" />

[details](https://medium.com/swlh/building-a-neural-graph-based-dependency-parser-f54fb0fbbf8d)





