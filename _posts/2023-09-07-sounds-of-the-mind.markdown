---
layout: post
title:  "Sounds of the Mind: Investigating Music's Role in Mental Health"

date:   2023-09-07 1:00:00 -0400
categories: blog
---

## Introducing the problem

In a time where music is readily available and saturates our daily lives, its
influence on our mental health has become an intriguing and pressing question.
In this report, my mission is to explore how music can both positively and
negatively impact mental health. I aim to address several pivotal questions:
Which music genres are associated with self-reported anxiety, depression,
insomnia and OCD? Can the number of hours spent listening to music each day
exert an influence on one's mental health? Moreover, does the tempo of one's
favorite genre hold the potential to positively affect mental health? By
exploring these questions, we aim to understand how the music we love affects
our mental well-being. Our findings can benefit both individuals looking to
music as a form of therapy and mental health research as a whole.

## Introducing the data

For this project, I have selected the dataset titled 'Music & Mental Health
Survey Results,' which I obtained from Kaggle. This dataset, comprising 33
columns and 736 rows, collected survey responses through a Google Form, with
respondents of varying ages and locations. It includes variables related to
individuals' music habits and their self-reported levels of anxiety,
depression, insomnia, OCD, and overall mental health. Below is a small preview
of what the dataset has to offer using Pandas:
```
import pandas as pd
mxmh = pd.read_csv("mxmh_survey_results.csv")
mxmh.head()
```
![1](/assets/p1/1.png)

## Pre-processing the data
To begin data cleaning, the first step is to identify columns containing null
values that require attention. Leaving null values in a dataset could result in
inaccurate calculations or conclusions. Utilizing Pandas, I listed out columns
with null counts and then displayed any columns with a count greater than zero:
```
null_counts = mxmh.isna().sum()
null_counts[null_counts > 0]
```
![2](/assets/p1/2.png)

Among these columns, I will only focus on BPM and Music Effects for the study.
But with the dataset being 736 rows, I can confidently remove any row
containing null data without losing too much valuable data:
```
mxmh_no_nulls = mxmh.dropna(axis=0)
```

With nulls taken care of, I need to lastly check for outliers. I used Pandas to
generate a descriptive summary of the data:

```
mxmh_no_nulls.describe()
```
![3](/assets/p1/3.png)

Looking at the results of this summary, I decided to filter out the following:

1) Rows where the hours per day listened is zero. They do not apply to this dataset.

2) Rows where the hours per day is greater than 16, because that is the average
awake hours of a human in a day.

3) Rows where the BPM is over 1000, because it is extremely rare to find a
genre with that high of a tempo.

```
mxmh_no_nulls = mxmh_no_nulls[mxmh_no_nulls['BPM'] <= 1000]
mxmh_no_nulls = mxmh_no_nulls[mxmh_no_nulls['Hours per day'] <= 16]
mxmh_no_nulls = mxmh_no_nulls[mxmh_no_nulls['Hours per day'] >= 0]

mxmh_no_nulls.describe()
```
![4](/assets/p1/4.png)

Now, I can move on knowing that the outliers are taken care of.

## Data Understanding/Visualization
For all these visualizations, I used matplotlib to create charts.

For the first visualization, I will be answering: Which music genres are
associated with self-reported anxiety, depression, insomnia and OCD?

![Average Values by Favorite Genre](/assets/p1/average_values_by_favorite_genre.png)

Below are the top 3 genres for each mental health condition:

Anxiety: Folk, K Pop, and Video Game Music.

Depression: Lofi, Hip Hop, and Rock.

Insomnia: Gospel, Lofi, and Video Game Music.

OCD: Lofi, Rap, and Pop.

Much of the data analyzed showed similar patterns, but there were some
intriguing findings. For instance, K-Pop stood out as being strongly associated
with anxiety compared to other conditions. On the other hand, Gospel music
seemed to have low associations with OCD but a higher correlation with
insomnia. Lastly, Lofi music was unique in reporting more cases of
self-reported depression than anxiety.

For our second visualization, I will be answering: Can the number of hours
spent listening to music each day exert an influence on one’s mental health?

![Average Hours per Day by Music Effects](/assets/p1/average_hours_per_day_by_music_effects.png)

While this chart may appear straightforward, it provides a clear response to
our initial inquiry. It shows that as the number of hours per day spent
listening to music increases, there is a noticeable improvement in the
prevalence of positive mental health conditions.

For our third visualization, I will be answering: Does the tempo of one’s
favorite genre hold the potential to positively affect mental health?

![Average BPM by Music Effects](/assets/p1/average_bpm_by_music_effects.png)

This chart is also quite straightforward, and though the differences are
subtle, it does indicate that upbeat music has a positive impact on improving
mental health conditions.


## Storytelling
My analysis reveals some associations between music genres and mental health
conditions, such as K Pop's link with anxiety and Gospel's connection with
insomnia. We found that increased daily music consumption correlated with
improved mental health. And lastly, while the influence of tempo was subtle,
upbeat music showed a positive impact. These findings answer the initial
questions and emphasize certain music's potential role in enhancing mental
well-being.

## Impact
This project holds the potential to make a positive impact by shedding light on
the relationship between music and mental health. However, there are possible
implications and limitations. While the visualizations provide insights, they
do not establish causation between music and mental health conditions.
Additionally, the dataset is based on self-reported survey responses, which may
have introduced bias. Moreover, the data does not factor in any individual life
experiences or backgrounds that could cause these conditions.

In summary, the project holds valuable insights, but it is essential to
approach the findings with caution and consider the overall context of
individual experiences to fully correlate mental health conditions with music.

## References
[Kaggle Dataset - Music & Mental Health Survey Results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)

## Code
You can view all the code for this project [here](/assets/p1/project_one_code.html).
