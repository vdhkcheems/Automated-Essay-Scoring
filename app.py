from flask import Flask, request, render_template
import pickle
import joblib
from imp import *
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    essay = request.form['essay']
    cleanessay = dataPreprocessing(essay)
    dic = {'essay_id' : [1],
        'full_text' : [essay],
        'cleaned_essay_text' : [cleanessay]}
    data = pd.DataFrame(dic, index=[0])
    data['word_count'], data['avg_word_length'], data['spell_error'], data['sent_count'], data['avg_sent_length'], data['para_count'], data['avg_para_length'] = zip(*data['full_text'].apply(extract_features))
    
    data['textstat_features'] = data['cleaned_essay_text'].apply(textstat_features)
    train_textstat = pd.DataFrame(data['textstat_features'].tolist())

    # Ensure the indices are unique
    data = data.reset_index(drop=True)
    train_textstat = train_textstat.reset_index(drop=True)

    #making final dataset
    final_data = pd.concat([data, train_textstat], axis=1)
    final_data = final_data.drop(columns=["textstat_features", "full_text","essay_id"])

    final_data["tokenized_text"] = final_data["cleaned_essay_text"].apply(word_tokenize)


    # Specify the path to your saved model
    model_path = 'word2vec.pkl'

# Load the model
    w2v_model = joblib.load(model_path)


    vect = np.array([get_avg_w2v_vector(essay, w2v_model) for essay in final_data["tokenized_text"]])

    additional_features = np.array(final_data.drop(columns=['cleaned_essay_text', 'tokenized_text']))
    X = np.hstack((vect, additional_features))

    with open('models_lgbm.pkl', 'rb') as f:
        models_lgbm = pickle.load(f)

    # with open('models_xgb.pkl', 'rb') as f:
    #     models_xgb = pickle.load(f)
    
    probabilities_lgbm = []
    for model in models_lgbm:
        # Predict the probabilities for the test features using the selected features
        proba_lgbm = model.predict(X, num_iteration=model.best_iteration)
        probabilities_lgbm.append(proba_lgbm)

    predictions_lgbm = np.mean(probabilities_lgbm, axis=0)

    # probabilities_xgb = []
    # for model in models_xgb:
    #     # Predict the probabilities for the test features using the selected features
    #     proba_xgb = model.predict(X)
    #     probabilities_xgb.append(proba_lgbm)
    
    # predictions_xgb = np.mean(probabilities_xgb, axis=0)
    # prediction_final = np.array((predictions_lgbm, predictions_xgb))
    # prediction_final = np.mean(prediction_final, axis=0)
    prediction_final_clip = np.round(predictions_lgbm.clip(0, 10))

    # Predict the score
    score = prediction_final_clip[0]
    return render_template('score.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)