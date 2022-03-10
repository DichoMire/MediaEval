# MediaEval

The MediaEval 2015 Workshop took place in Wurzen, Germany, on 14-15 September 2015 as a satellite event of Interspeech 2015.

One of the many challenges was to create a machine learning model to predict whether or not a tweet is spreading false news.

For my solution I am using a variation of Multinomial Naive Bayes, with additional preprocessing of the data.
The preprocessing includes removing of:
  - whitespace
  - web links
  - emojis
  - large quantity of rare UTF-8 characters which aren't prevalent in this dataset
  - stopwords

The result is an algorithm, which performs exceptionally well:

<img width="373" alt="tmep" src="https://user-images.githubusercontent.com/56360395/157566394-81929a77-8cd8-40ca-9df4-27a1b7bac4a6.png">
