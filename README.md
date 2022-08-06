# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Subreddit Posts Extraction & Classification

## Background

[*Twitch*](https://www.twitch.tv/p/en/about/) is an American interactive livestreaming service for content spanning gaming, entertainment, sports, music, and more.

Similar to many other online social platforms, we are always looking for new ways to better engage our users, expands our products and service offerings, thereby increasing customer stickiness through greater user experience and improving both top and bottom lines.

_**Why Gaming?**_<br>
- **Video Gaming Industry**: ~$178.73 Billion in 2021 (increase of 14.4% from 2020) ([*source*](https://www.wepc.com/news/video-game-statistics/))

- **Global eSports market**: ~$1.08 Billion in 2021 (increase of ~50% from 2020) ([*source*](https://www.statista.com/statistics/490522/global-esports-market-revenue/))

- **eSports industry's global market revenue**: Forecasted to grow to as much as $1.62 Billion in 2024. 

- China alone accounts for almost 1/5 of this market. 

In recent months, we started a pilot program with a subset of our most active users by availing them to a new beta forum that has sparked many discussions amongst our gaming users.

This has resulted in hightened traffic with frequent posts and comments updates daily. Our business development and marketing counterparts also realised these gaming users are predominantly focusing on 2 games, namely [***Dota 2***](https://www.dota2.com/home) and [***League of Legends (LoL)***](https://leagueoflegends.com). 

Our business development and marketing colleagues see great potential in tapping on this group of active gamers and the associated data. However, since there is merely 1 single beta gaming forum thread, users have to sieve through multiple posts to find topics that interest or are relevant to them, resulting in potential poor user experience. Additionally, it would be more effective and efficient to target each game's user base separately by designing sales and marketing campaigns that better meet the corresponding user base's needs.

---

## Problem Statement

- Our business development and marketing colleagues have requested for us, the Data Science team, to design an **AI model** that **correctly classifies posts** in the 1 single beta gaming forum thread into 2 separate threads, 1 for Dota 2 and another for League of Legends (LoL), with an **accuracy of at least 85%** and **Top 10 Predictors for each subreddit** thereby improving user experience and increasing ease of designing more targeted sales and marketing campaigns that better meet the corresponding user base's needs.

**Datasets scraped:**
   - **`dota2_raw.csv`**: Dota 2 dataset
   - **`lol_raw.csv`**: League of Legends (LoL) dataset
<br>

**Brief Description of Datasets selected:** 
- The 2 datasets above, each comprising 4,000 records, were scrapped with the [*pushshift API*](https://github.com/pushshift/api) from these 2 subreddits : 
    - [***r/DotA2***](https://www.reddit.com/r/DotA2/); and 
    - [***r/leagueoflegends***](https://www.reddit.com/r/leagueoflegends/)

In this project, I scraped data from 2 subreddits, cleaned and preprocessed them for visualisation. Subsequently, I tested 3 models using 2 vectorizers and determined that Multinomial Naive Bayes using TF-IDF vectorizer produces the highest accuracy.

---

### Baseline Score for Accuracy: 0.5868

We established **0.5868** as the **baseline score** for our best model to beat.

This baseline score was derived by adopting a highly simplistic approach of predicting every post to be a Dota2 post. Since there are **3414** Dota2 posts and **2404** League of Legends posts, corresponding to **58.68%** and **41.32%** of total posts respectively after removing duplicates and moderated/deleted posts, we would have scored **0.5868** (i.e. **58.68%**) in terms of **Accuracy**.

As there are approximately equal number of observations in each class, and all predictions and prediction errors are equally important, a classification model evaluation metric appropriate for our purpose is **Accuracy**.

---

## Data Cleaning, Preprocessing and Visualisation

We performed the following steps for data cleaning, preprocessing and visualisation:
1. Import relevant libraries
2. Data Cleaning (Part 1):
   - Check for Duplicates
   - Delete Rows for Removed Posts
3. Establish Baseline Score for Accuracy
4. Data Cleaning (Part 2) & Preprocessing:
   - Replace null cells with ''
   - Replace URLs in text strings with ''
   - Replace Punctuations in text strings with ' '
   - Tokenize Data
   - Remove Tokenized Words with customised Stopwords List 
   - Data Lemmatization
   - Data Stemming
   - Sentence Forming with Processed Tokens
5. Visualising Most Frequent Words
   - Count Vectorization - 1 & 2 ngram Count Vectorizer
   - TF-IDF Vectorization - 1 & 2 ngram Count Vectorizer

---

### Visualising Most Frequent Words 

After data cleaning is completed, we visualised the 25 most frequent words (1 ngram) in the posts for both subreddits.

![Top 25 Most Frequent Words (1 ngram)](media/Top_25_Most_Frequent_Words_(1_ngram).png)

We repeated the visualisation for the 25 most frequent pair of words (2 ngram) in the posts from both subreddits.

![Top 25 Most Frequent Words (2 ngram)](media/Top_25_Most_Frequent_Words_(2_ngram).png)

- Many of the words in the 1 ngram most frequent words list also appeared in the 2 ngram visualisation.
- Generally, 2 ngram results provide greater details on the context of the highest frequency words in each subreddit than 1 ngram.
- Comparing the words for each ngram across the 2 vectorizers used, relative importance of most words have changed. This is because the count vectorizer merely focuses on the frequency of words present in the corpus while TF-IDF vectorizer also provides the importance of the words in addition to frequency of words present

---

### Model Testing

1. We will perform hyperparameter tuning on these 3 models:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest Classifier

   using these 2 vectorizers:
    - Count vectorizer
    - TF-IDF vectorizer
<BR>
<BR>    
2. For each model, we will also determine the following:
    - Optimal Grid Search Model Parameters
    - Optimal Grid Search Best Score
    - Train AUC Score
    - Test AUC SCore
    - Train Accuracy Score
    - Test Accuracy Sore
    - F1-Score
    - Precision
    - Recall or Sensitivcity
    - Specificity

   and chart the following:
    - Confusion Matrix
    - Receiver Operating Characteristic (ROC) Curve
<BR>
<BR>    
3. We will also display the top 10 predictors of each model.
    
4. Finally, we will summarise model performance, analyse and determine the most appropriate model for this dataset to address the problem statement. 

---

### Summary Table of Model Performance

|                     **Model** | **Logistic Regression** | **Logistic Regression** | **Multinomial Naive Bayes** | **Multinomial Naive Bayes** | **Random Forest Classifier** | **Random Forest Classifier** |
|------------------------------:|------------------------:|------------------------:|----------------------------:|----------------------------:|-----------------------------:|-----------------------------:|
|                **Vectorizer** |               **Count** |              **TF-IDF** |                   **Count** |                  **TF-IDF** |                    **Count** |                   **TF-IDF** |
|        Grid Search Best Score |                0.949473 |                0.958786 |                    0.962992 |                    0.964984 |                     0.931221 |                     0.936600 |
|               Train AUC Score |                0.989201 |                0.994599 |                    0.984728 |                    0.993497 |                     0.998261 |                     0.998971 |
|          Validation AUC Score |                0.948869 |                0.958723 |                    0.962233 |                    0.963496 |                     0.937692 |                     0.943956 |
|          Train Accuracy Score |                0.904715 |                0.942534 |                    0.938605 |                    0.960707 |                     0.994843 |                     0.994843 |
| **Validation Accuracy Score** |                0.860825 |                0.882016 |                    0.891753 |                **0.902062** |                     0.873425 |                     0.891180 |
|                      F1-Score |                0.892144 |                0.905418 |                    0.906759 |                    0.918610 |                     0.893494 |                     0.912281 |
|                     Precision |                0.818404 |                0.855160 |                    0.917166 |                    0.896840 |                     0.882857 |                     0.865907 |
|         Recall or Sensitivity |                0.980488 |                0.961951 |                    0.896585 |                    0.941463 |                     0.904390 |                     0.963902 |
|                   Specificity |                0.690707 |                0.768377 |                    0.884882 |                    0.846047 |                     0.829404 |                     0.787795 |
|            True Negative (TN) |                     498 |                     554 |                         638 |                         610 |                          598 |                          568 |
|           False Positive (FP) |                     223 |                     167 |                          83 |                         111 |                          123 |                          153 |
|           False Negative (FN) |                      20 |                      39 |                         106 |                          60 |                           98 |                           37 |
|            True Positive (TP) |                    1005 |                     986 |                         919 |                         965 |                          927 |                          988 |

---

### Multinomial Naive Bayes' (TF-IDF vectorizer) Score for Accuracy: 0.9021

When assessing the suitability of the Multinomial Naive Bayes (TF-IDF vectorizer), we obtained the following Confusion Matrix and Receiver Operating Characteristic (ROC) results:

![CM & ROC (NB TF-IDF Vectorizer)](media/CM_&_ROC_(NB_TF-IDF_Vectorizer).png)

The corresponding top 10 predictors for each subreddit are as follows:

![Top 10 (NB TF-IDF Vectorizer)](media/Top_10_(NB_TF-IDF_Vectorizer).png)

When evaluating each model and its associated vectorizer previously, we noted that for both Logistic Regression and Multinomial Naive Bayes models using TF-IDF Vectoroizer, **ALL** of the above top 10 predictors for each subreddit are also found amongst the top 25 most frequent words (1 ngram) for **both** subreddit.

The logic here is that if **ALL** the top 10 predictors of the best model also appear amongst the top 25 most frequent words (1 ngram) for **both** subreddit, it is more likely for at least 1 of these predictors to appear in a post to be classified by the best model.

Referencing the summary table of model performance above, **Multinomial Naive Bayes using TF-IDF vectorizer** yields the highest **Test Accuracy Score** of **0.902062** which decisively beats the Baseline Score of **0.5868**. Additionally, since the Test Accuracy Score is worse than the Train Accuracy Score by merely 0.058645, there is likely no over-fitting issues.

Thus, Multinomial Naive Bayes' (TF-IDF vectorizer) is chosen as the best model for our dataset.

---

### Limitations and Future Plans

**1) Model Accuracy Improvement**
    - Better data cleaning steps e.g. remove additional Character Entities
    - Further hyperparameter tuning to enhance accuracy, while balancing potential risks of overfitting ([*source*](https://www.tmwr.org/tuning.html))
    - Identify and rectify cause(s) for model's post misclassifications (i.e. False Positives and False Negatives)

**2) Include New Model Features**
    - Analyse moderated posts to implement auto regulation of potential user rogue behaviour e.g. profanities, spams, scams, etc.
    - Sentiment analysis

---

### Recommendations and Conclusion

_**Internal**_
1) Roll out AI model classification to split posts into 2 separate threads
    - Multinomial Naive Bayes' (TF-IDF vectorizer) Model accuracy > 85% and Top 10 Predictors for each subreddit have been identified
  
2) Establish timeline for future model features roll-out, e.g. auto Twitch forum moderation, sentiment analysis, etc.

3) Tease out campaign specific insights e.g. new users acquisition, advertising, etc.

<br>

_**External**_
1) **Developers**:
   - Data Insights as a Service on sentiments, new games &/or features, etc.
   - Explore potential advertising, partnership, sponsorship opportunities, etc.
 
2) **Esports/Events**:
   - Data Insights as a Service on game interests, event/tournament mechanics, etc.
   - Explore potential advertising, partnership, sponsorship opportunities, etc.

3) **Gaming Streamers**:
   - Insights on peak online user activities, etc.

In conclusion, I developed a Multinomial Naive Bayes' (TF-IDF vectorizer) Model with accuracy > 85% and Top 10 Predictors for each subreddit have been identified, thereby meeting the objectives set out by the Problem Statement.

With the blessings from the business development and marketing stakeholders, I shall proceed to implement the items set out above in the limitations, future plans and recommendations sections. I am confident that we could potentially further enhance user experience when the posts are eventually split into 2 distinct threads. I also look forward to partnering all of you in teasing out campaign specific insights to improve both top and bottom lines for the organisation!