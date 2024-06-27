Project Title: Comparative Sentiment Analysis and Prediction Accuracy of App Reviews Using Machine Learning and Deep Learning

In this machine learning project, I compared two apps by analyzing their user reviews' sentiments and predicting sentiment accuracy using various ML and deep learning approaches.

Reviews were classified into sentiments (positive, negative, neutral) on over 1.5 lakh user reviews of 2 apps using unsupervised methods like TextBlob and VADER.

I compared both apps by examining the number of words and characters for positive and negative sentiments, the top 40 positive and negative review word clouds for unigrams and bigrams, and the percentage distribution of sentiments to understand user feedback differences.

The combined reviews from both apps were split into training and testing sets for model evaluation.

Performed feature extraction with TF-IDF and Word2Vec embeddings, hyperparameter tuning using GridSearchCV, and 7-fold Cross-Validation

Using TF-IDF vectorizer and Word2Vec embeddings separately, I trained 6 different ML models, including SVM, Multinomial Naive Bayes, Gaussian Naive Bayes, XGBoost, Random Forest, and Logistic Regression, to predict review sentiments.

I also employed deep learning models, including CNN and LSTM, to compare their performance with the ML models.

Achieved 93.39% accuracy with Bidirectional LSTM using binary cross-entropy loss and Nadam optimizer in TensorFlow. Plotted AUC, Recall, F1, and Precision scores

The models' performance was evaluated, and the best-performing model was saved for future sentiment prediction on new reviews.

This project showcased how unsupervised sentiment analysis, combined with ML and deep learning, can effectively compare and predict sentiments in app reviews.

