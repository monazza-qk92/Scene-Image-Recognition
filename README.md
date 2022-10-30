# Scene-Image-Recognition
CS-867 Computer Vision Assignment 3 Spring 2021

Scene Classification is a task in which scenes from photographs are categorically classified. Unlike object classification, which focuses on classifying prominent objects in the foreground, Scene Classification uses the layout of objects within the scene, in addition to the ambient context, for classification.It is widely used for computer vision tasks like face recognition, face detection, video object.

In this project I used clustering and bag of words for scene recognition.

The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. The bag-of-words model has also been used for computer vision. In practice, the Bag-of-words model is mainly used as a tool of feature generation. After transforming the text into a "bag of words", we can calculate various measures to characterize the text. The most common type of characteristics, or features calculated from the Bag-of-words model is term frequency, namely, the number of times a term appears in the text. 

Lets begin with a few introductory concepts required Bag of words. We shall cover 4 parts (so keep scrolling !)

1.Clustering
2.Bag of Visual Words Model
3.Generating Vocabulary
4.Training and testing



Clustering : Lets say there is a bunch of Wrigleys Skittles. And someone is to tell you to group them according to their color. It’s quite simple .. aint it! Simply seperate all red, blue, green, etc in different parts of the room. Here, we differentiated and seperated them on basis of color only.
So moving on to a more complex situation that would give a much profound meaning to clustering. Suppose there is a room full of utilities, be it accesories, clothing, utensils, electronics, etc. Now, if someone is told to seperate out into well formed groups of similar items, one would essentially be performing clustering.

So yes, clustering can be said as the grouping a set of objects in such a way that objects in the same group are much similar, than to those in other groups/sets

Moving on, lets’ decide as to how we perform clustering. The selection of clustering algorithm depends more on what kind of similarity model is to be chosen. There are cases wherein, the plain’ol clustering impression that everyone so simply elucidates may not be the right choice. For example, there exists various models, such as centroid oriented - Kmeans, or Distribution based models - that involve clustering for statistical data; such places require Density based clustering (DBSCAN) , etc.
