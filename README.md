# Automated-Essay-Scoring
This repository is a personal project of mine making an automated essay scoring model using datasets from kaggle on lgbm and xgboost models. The model takes the essay and then scores it on a scale of 0 to 10.

# What I did?
There have been two competitions on kaggle regarding Automated Essay scoring, this repo has my implementation of that using both the datasets:
1. Firstly, I joined both the datasets together and then cleaned the essay text by converting everything to lowercase, removing punctuation, removing html tags and expanding contractions like changing "can't" to "cannot" and much more.
2. Then I performed feature extraction with a few self implemented features and a few textstat library features to create my final dataset.
3. Then, I vectorized the cleaned essay text using word2vec to convert the text to 1500 long vector by taking average of the vectors produced by each word and integrated it with my processed data to make the final dataset to train the model on.
4. I used stratifiedKfold to train the model and predict on test set to get a more robust evaluation metric and I used it on the two best performing models I could find, being Ligh Gradient Boosting Machine(LGBM) and eXtreme Gradient Boosting(XGB).
5.  I used cohen kappa score as the performance measuring metric for this project. The LGBM model had the cohen score of 0.806.. and the XGB showed a score of 0.803.. while their ensemble which is just the average of their prediction showed a score of 0.804.

# My Learnings
This is my first time working with Natural Language Processing and this turned out to be a great learning experience for me, teaching me about how computer understands human laguage and how it can work on it. Learning about different types of vectorizers, tokenizers, machine learning models has significantly improved my knowledge of NLP.

# Extra
I made a basic web app using flask for the project above. some screenshots of the web app are attached below:

![Screenshot_20240613_144131](https://github.com/vdhkcheems/Automated-Essay-Scoring/assets/136262842/8bedc50f-c47d-45a4-afe2-c5efc22ab85f)


![Screenshot_20240613_144153](https://github.com/vdhkcheems/Automated-Essay-Scoring/assets/136262842/a2f53013-597e-45a9-8b7b-d59578a4d68b)


![Screenshot_20240613_144243](https://github.com/vdhkcheems/Automated-Essay-Scoring/assets/136262842/748f1cd1-7556-4afd-8cb0-2c3a8e3ec521)


![Screenshot_20240613_144223](https://github.com/vdhkcheems/Automated-Essay-Scoring/assets/136262842/6302c850-7500-4d53-b502-5e3c85211321)
