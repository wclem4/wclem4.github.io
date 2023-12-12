---
layout: post
title: Sentiment Analysis
date: 2023-12-11 04:00:00 -0400
categories: blog
---
## Introduction

Sentiment analysis is a type of text classification that analyzes a piece of
text and determines whether the sentiment of it is positive or negative (1).
For this project, I will be striving to better understand public sentiment by
training a model which to determine the sentiment of inputted text. I would
like to know what kind of words might influence sentiment the most, and how
well a model can predict the sentiment of test data.

## About the Data

To train my model, I will be using the "sentiment140" dataset from Kaggle (2).
This dataset contains 1.6 million tweets that have been annotated with their
sentiment. The dataset contains a total of six features:
- sentiment (0 = negative, 4 = positive)
- id of the tweet
- date of the tweet
- flag
- user
- text of the tweet

For my model, I will only be using the sentiment and text features, because the
other features does not affect the results of sentiment analysis.
```
data = pd.read_csv('tweets.csv', encoding='latin-1', header=None, names=['sentiment', 'id', 'date', 'query', 'user', 'tweet'])
data = data.drop(columns=["id", "date", "query", "user"])
```

Below is a snippet of the first 20 features, of which we can see our sentiment
and tweet columns. There is a lot of cleanup to do with the tweets to ensure
this model performs as intended.
```
data.head(20)
```

<p align="center">
    <img src="/assets/final/head.png" width="700">
</p>

To demonstrate this is a balanced dataset, take a look at the Sentiment Distribution below:

<p align="center">
    <img src="/assets/final/dist.png" width="700">
</p>

## Methods

#### Baseline Model

Before making any changes to the tweets column, I decided to build a baseline
model on the untouched data to compare performance. I chose to train the data on
80% of the data, and run a test on the last 20%.
```
X = data['tweet']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

I then had to utilize the CountVectorizer from sklearn to convert all the text
into a numerical format for the Model to understand.
```
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
```

Lastly, I fit the training data to the Naive Bayes algorithm and predicted the test data.
```
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
y_pred = clf.predict(X_test_counts)
```

The evaluation of this run is shown in the next section; Evaluation.

#### Pre-Processing Steps

For the next model, I wanted to transform the tweets column to see if I could
improve on the results.

Looking at some of the first tweets, there are a few things I want to address.
To do this, I setup a lambda function named `preprocess_text` to run on each
tweet in the dataset.
```
data['tweet'] = data['tweet'].apply(lambda x: preprocess_text(x))
```

Inside this function, I first wanted to remove any tags and URLs in the tweets,
because the model will not be establishing sentiment based on the URLs or the
tagged account. To do this, I wrote a function for both tags and URLs that uses
a regex pattern (3) to remove any values that might match. Please see the code
attached at the bottom of this report to view the full implementation of these
functions.
```
text = remove_urls(text)
text = remove_tags(text)
```

I then built a python dictionary of some slang abbreviations (e.g. lol =
laughing out loud, omg = oh my god) I could think of that might affect a
sentiment reading, and a list of expanded contractions (e.g. can't = cannot,
don't = do not) I sourced online (4).
```
text = expand_abbreviations(text)
```

I then also removed the extra characters that are not in the alphabet or letters
0-9, because those will also not affect sentiment.
```
text = remove_extras(text)
```

Then, this left us with a much cleaner tweet column. But there was one more
thing I wanted to include. Termed "Stop Words", they are commonly used words
that are fillers in a sentence (like "the", "a", "an") that search engines are
built to ignore (5) because they do not affect the meaning of the sentence, only
assist readability. This is done pretty easily with the `stopwords` list pulled
from the NLTK python library.
```
words = text.split()
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
preprocessed_text = ' '.join(filtered_words)
```

#### Revisiting the Data (scrubbed edition)

Now that we have the URLs, tags, abbreviations, extra characters, and filler
words removed, I wanted to see some of the more common words that come from
the positive and negative sentiments.

I used the Counter tool from the collections python module to total up the word
counts.
```
positive_text = data[data['sentiment'] == 4]['tweet']
positive_words = ' '.join(positive_text).split()
positive_word_counts = Counter(positive_words)

most_common_positive_words = positive_word_counts.most_common(20)
```

Printing these out, I get the following as the most common positive words:
```
1. good: 60858 times
2. love: 46848 times
3. day: 44372 times
4. laugh: 37492 times
5. like: 37109 times
6. get: 36172 times
7. loud: 35397 times
8. thanks: 33747 times
9. going: 30648 times
10. u: 30211 times
11. time: 29339 times
12. today: 28420 times
13. go: 27855 times
14. got: 27841 times
15. new: 26642 times
16. know: 25978 times
17. one: 25831 times
18. see: 25464 times
19. great: 24884 times
20. back: 23655 times
```

And these as the most common negative words:
```
1. get: 45318 times
2. go: 45177 times
3. work: 44015 times
4. like: 40640 times
5. day: 38005 times
6. today: 36199 times
7. going: 33443 times
8. cannot: 33081 times
9. got: 33039 times
10. back: 32618 times
11. really: 31224 times
12. miss: 30502 times
13. im: 30353 times
14. want: 29734 times
15. oh: 29543 times
16. still: 28760 times
17. good: 28553 times
18. know: 27722 times
19. sad: 27147 times
20. time: 26772 times
```

There is a clear separation between these two lists, and it is very easy to
understand why majority of these words are regarded as positive or negative.
This answers my initial question on what words might influence sentiment the
most.

#### New Model

Finally, I trained a new model to see how it performs with the changed data. I
used the exact same parameters, algorithm, and steps as the baseline model at
the start of this section.

## Evaluation

To evaluate my models, I chose to print out the 4 main accuracy metrics for each model:
- Accuracy: Ratio of correct instances to total instances
- Precision: Proportion of correctly identified positive cases out of all positive predictions.
- Recall: Proportion of true positive predictions out of all positive results.
- F1 Score: Harmonic mean of Precision and Recall.

For the baseline model, I had the following results:

```
Accuracy: 0.7809
Precision: 0.7827
Recall: 0.7809
F1 Score: 0.7806
```

For the pre-processed model, I had the following results:
```
Accuracy: 0.7676
Precision: 0.7681
Recall: 0.7676
F1 Score: 0.7675
```

Surprisingly, the original model was already almost 80% accurate, so
improvements from there were going to be tough to do. Unfortunately my
pre-processing did not improve on the baseline model, which could be for
several reasons I will discuss in the conclusion.

## Storytelling & Conclusion

Through this project, I was able to deep dive into what makes text result in a
positive or negative sentiment. I was able to achieve my goals of building a
successful model that will correctly predict the sentiment of text almost 80% of
the time, and I found which words influence the sentiment of a message the most.

The only goal I fell short on is improving on my baseline model. I made several
modifications to the messages, including removing URLs and tags, expanding
abbreviations, removing special characters, and removing stop words. This was
all in hopes to give the model a simpler set of data to work with, and prevent
extra words from negatively influencing the results.

One thing I could have improved on is building a better dictionary for the
abbreviations and contractions. I am sure there are plenty more acronyms and
contractions I had missed out on expanding. Unfortunately I could not find any
python libraries with a dictionary that included that, so it had to be created
manually.

Another potential setback could be figurative language, like sarcasm for
example. Sarcasm can be hidden in plain text to a model, but easily understood
by a human. Many sarcastic statements can include similar text to the opposite
sentiment, and that can be very difficult to track.

For some future steps, I might try and pull the abbreviations from an API with
a much greater list, because there are no true python libraries available to do
so.

## Impact

The impact of this sentiment analysis has various social and ethical implications.

Socially, a model like this can be used by businesses and organizations to view
how their product or idea is perceived by the general public. It can also be
used for predicting the mental health of individuals based on what they might
post on social media.

Ethically, this can raise concerns about privacy. Running the model on one's
personal thoughts and ideas they put on social media without consent could be a
breach of their privacy. Being from a public dataset, there could also be biases
in the training data that might lead to unfair or discriminatory results.

## References

1 [Sentiment Analysis Definition: towardsdatascience.com](https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17)

2 [Dataset: kaggle.com](https://www.kaggle.com/datasets/kazanova/sentiment140/data)

3 [URL Regex: urlregex.com](https://urlregex.com/index.html)

4 [Contractions List: sjsu.edu](https://www.sjsu.edu/writingcenter/docs/handouts/Contractions.pdf)

5 [Stop Words Definition and Python Tutorial: geeksforgeeks.org](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)

## Code

You can view all the code for this project [here](/assets/final/final.html).
