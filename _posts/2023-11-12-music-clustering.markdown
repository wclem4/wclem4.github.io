---
layout: post
title:  "Clustering the Top Songs of 2023"

date:   2023-11-12 1:00:00 -0400
categories: blog
---

## Introducing the Problem

Music genres are labels that group songs with similar patterns and styles.
Traditionally they are based on features like instrumentation, rhythm, and
themes. For this project, I will be using clustering to algorithmically divide
up songs into their own genres based on a list of unique features.

The problem I am trying to solve for this project is: How can clustering be
used to split a dataset of music into hypothetical genres?

I would also like to answer: What features of a song most influences its
streaming numbers?

## What is clustering? How does it work?

Clustering is the idea of getting an algorithm to "cluster" data points
together without any user supervision.

K Means is a popular clustering algorithm I will be using. It partitions the
dataset into K clusters by assigning data points to centroids and updating the
centroids until all the clusters stabilize. Below are the steps it goes through
to cluster a dataset.

1. Choose K value, aka the number of clusters
2. Initialize the centroids of each cluster
3. Assign every point in dataset to the cluster with closest centroid
4. Update centroid value to the center of all the points assigned to it
5. Repeat steps 3 & 4 until the centroids stop shifting

## Introducing the Dataset

The dataset I chose to use was sourced from Kaggle and titled [Most Streamed
Spotify Songs
2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023),
and I've listed all the features I found most useful below:

- `streams` - number of streams on Spotify
- `bpm` - beats per minute of the song
- `key` - key of the song
- `mode` - major or minor mode of the song
- `danceability_%` - how suitable the song is for dancing
- `valence_%` - positivity level of the song
- `energy_%` - energy level of the song
- `acousticness_%` - amount of acoustic content in the song
- `instrumentalness_%` - amount of instrumentation in the song
- `liveness_%` - presence of live performance elements in the song
- `speechiness_%` - amount of spoken words in the song

Below gives a small preview of the values from these features:
```
# read in data
raw_data = pd.read_csv('spotify-2023.csv', encoding='latin-1') # need Latin encoding to parse this dataset
raw_data.head()
```
<p align="center">
    <img src="/assets/p4/head.png" width="700">
</p>

## Data Understanding

To better understand the data, I chose to build some visualizations. First, I
built a correlation heatmap to see which features might correlate closest or
furthest from each other.

The most notable feature is Energy, which is highly correlated with Valence and
Danceability. This is not too surprising, as a high energy song usually will
bring out some positive values and offer a suitable dancing beat. On the other
hand, energy and acoustic-ness do not correlate at all, which is also to be
expected because acoustic acts lack a full band.

There are several other interesting observations from this heatmap that are
less impactful, but still notable.

<p align="center">
    <img src="/assets/p4/correlation_heatmap.png" width="700">
</p>
The second visualization I built is a set of pairplots that compare the number
of streams a song has to the rest of the features, which can answer my second
question. Because there are a lot of plots, I've made them accessible
[here](/assets/p4/streams_pairplots.png). Some interesting findings are that
more popular songs (higher streams) are played in a Major scale instead of a
Minor scale. It also appears that as song popularity increases, so do the songs
played in C#.

## Pre-processing

Before using the data in our model, I chose to filter the features of the
dataset down to just the important ones:

```
# filter down to just the important columns
important_features = ['streams', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
data = raw_data[important_features]
```

To better work with the key and mode columns, I chose to create dummies for
those so that we can individually compare each key and mode in the dataset:

```
# create dummies for the key and mode columns
key_dummies = pd.get_dummies(raw_data['key'], prefix='key')
mode_dummies = pd.get_dummies(raw_data['mode'], prefix='mode')
data = pd.concat([data, key_dummies, mode_dummies], axis=1)
```

Then, the streams column had to be converted away from an object type. I
converted all the rows to a numeric value, and filtered out any rows out that
might not include numeric values:
```
# convert streams column to a numeric type instead of object
data = data[pd.to_numeric(data['streams'], errors='coerce').notnull()]
data['streams'] = data['streams'].astype(int)
```

Lastly, I checked for nulls, which resulted in none:
```
# view null counts
data.isna().sum()
```

## Modeling

I chose to use the KMeans algorithm from sklearn for my model, as it is the
easiest to display and understand for this dataset.

I first dropped the streams column, because that will negatively affect our
results in splitting the dataset into genres:
```
# don't want to cluster based on streams, because that does not affect how the song sounds
X = data.drop(columns=['streams'])
```

I then applied Principal Component Analysis to the dataset to setup the data
for the model:
```
# apply principal component analysis
pca = PCA(n_components=4)
pca_fit = pca.fit_transform(X)
pca_df = pd.DataFrame(pca_fit)
```

I then utilized the elbow method to find the best K value for the dataset, then
plotted it:
```
# implement elbow method to find best K value
inertia = []
for k in range(1,12):
    kmeans = KMeans(n_clusters=k, random_state=1, n_init='auto').fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))
```
The elbow of the line tells me that 4 is best number of clusters, because that
is when the line stops dropping at a high rate:
<p align="center">
    <img src="/assets/p4/elbow_method.png" width="700">
</p>

Lastly, I ran the model and plotted the values on a scatterplot:
```
kmeans = KMeans(n_clusters=4, random_state=1, n_init='auto').fit(X)
y = kmeans.fit_predict(X)
sns.scatterplot(x = pca_df[0], y = pca_df[1], hue=y)
```
<p align="center">
    <img src="/assets/p4/kmeans_scatter.png" width="700">
</p>

As seen, the model nicely split the dataset into 4 clusters, separated by hue.

## Storytelling/Analysis

The clusters reveal distinct groups of songs based on the features in the
dataset, which could line up with conventional genre distinctions. This allows
us to identify common songs within each cluster; for instance Cluster 1 could
include more higher energy, danceable songs while Cluster 2 could include slow,
more acoustic-heavy tracks. Now instead of manually categorizing music, we can
run each song through an algorithm like this to tell us exactly what the genre
might be, and find other songs that are similar.

## Impact

The impact of this project extends beyond just clustering songs into groups. By
understanding how the attributes contribute to these clusters, it has the
potential to influence the way music is recommended and discovered by
listeners. This could improve user experience for streaming applications by
suggesting new tracks they might not have known about. Negatively, separation
of genres could cause users to miss out on new music that they might like.

## References
[Kaggle Data Set](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)

## Code
You can view all the code for this project [here](/assets/p4/p4.html)
