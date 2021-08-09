# Comparison of classifiers

- Repository: `challenge-clustering`
- Type of Challenge: `Learning`
- Duration: `3 days`
- Deadline: `dd/mm/yy H:i AM/PM`
- Team challenge : `individual`

## Learning Objectives

You will learn to implement different clustering algorithms in Python.
At the end of the challenge, you will be able to:

- Choose the most appropriate algorithm, depending on the problem.
- Know the know-how, way to implement and logic behind most common clustering techniques.
- Manipulate different types of data.

NOTE: This challenge is a continuation of the classification challenge 'Bearing analysis'. The learners should have extracted some features from the dataset already.

## The Mission

Your boss was happy with your previous work on the bearing analysis, guessing faulty bearings was a piece of cake for you! The team came back to you for further expertise;
They want to know what type of failures occur! Or rather, if the failures exhibit similarities to other failures. This is a perfect **clustering** challenge.

The [dataset](https://www.kaggle.com/isaienkov/bearing-classification?select=bearing_signals.csv) is still available, but I advise you to use the processed dataset from last assignment.

### Must-have features

- gathering of **features** (5+) of your **failed** bearing dataset
- kmeans clustering of at least two **features** of the **failed** bearings
- visualization of said clustering
- evaluation of the **silhouette** score of your clustering method
- extension (one-by-one) to 6 features. (how does the silhouette score evolve?)
- vizualizations of your model evaluation

### Nice-to-have features

- Hyperparameter tuning of kmeans method
- Hyperparameter tuning of  other (2+) clustering methods
- vizualization of other clustering evaluation metrics besides silhouette score
- comparison of different distance metrics used in your clustering methods (either through dataset transformation or changing the clustering cost function)

## Deliverables

1. Publish your source code on the GitHub repository.
2. The deployment link, if applicable.
3. Pimp up the README file:
   - Description
   - Installation
   - Usage
   - (Visuals)
   - (Contributors)
   - (Timeline)
   - (Personal situation)

### Steps

1. Create the repository.
2. Study the request (What & Why ?)
3. Identify technical challenges (How ?)
4. Implement different classifiers.
5. Compare the results to find the more convenient algorithm.

## Evaluation criteria

| Criteria       | Indicator                                                               | Yes/No |
| -------------- | ----------------------------------------------------------------------- | ------ |
| 1. Is complete | At least 1 clustering method was used.                             |        |
|                | Code re-use was limited using functions and classes.                    |        |
|                | There is a published GitHub repo with those.                            |        |
| 2. Is correct  | multiple models were compared between each other and a conclusion was drawn.  |        |
|                |  |        |
| 3. Is clean    | There is good documentation in the project.                             |        |
|                | The code is formatted nicely.                                           |        |

## A final note of encouragement

![You've got this!](https://media.giphy.com/media/3og0IFrHkIglEOg8Ba/giphy.gif)