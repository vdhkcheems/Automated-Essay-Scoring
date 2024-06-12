import numpy as np
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
import textstat
from spellchecker import SpellChecker
from sklearn.metrics import cohen_kappa_score


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def score_normalise(n):
    temp = (n*10)/6
    temp = round(temp)
    if temp < 0:
        temp = 0
    elif temp > 10:
        temp = 10
    else:
        temp = temp
    return temp

def textstat_features(text):
    features = {}
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['automated_readability_index'] = textstat.automated_readability_index(text)
    features['difficult_words'] = textstat.difficult_words(text)
    features['text_standard'] = textstat.text_standard(text, float_output=True)
    features['reading_time'] = textstat.reading_time(text)
    features['syllable_count'] = textstat.syllable_count(text)

    return features


def count_spelling_errors(text):
    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    return len(misspelled)

def sentence_count(text):
    sentence = 0
    for i in text:
        if i in ".?!":
            sentence += 1
    return sentence

def average_sentence_length(text):
    # Split the text into sentences using regular expressions
    sentences = re.split(r'[.!?]', text)
    
    # Filter out any empty sentences that may result from the split
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Calculate the total number of words in all sentences
    total_words = sum(len(sentence.split()) for sentence in sentences)
    
    # Calculate the average sentence length
    average_length = total_words / len(sentences) if sentences else 0
    
    return average_length

def count_paragraphs(text):
    # Split the text into paragraphs using newline characters
    paragraphs = text.split('\n')
    
    # Filter out any empty paragraphs that may result from the split
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    
    # Count the number of non-empty paragraphs
    num_paragraphs = len(paragraphs)
    
    return num_paragraphs

def average_paragraph_length(text):
    # Split the text into paragraphs using newline characters
    paragraphs = text.split('\n')
    
    # Filter out any empty paragraphs that may result from the split
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    
    # Calculate the total number of sentences in all paragraphs
    total_sentences = sum(sentence_count(paragraph) for paragraph in paragraphs)
    
    # Calculate the average paragraph length
    average_length = total_sentences / len(paragraphs) if paragraphs else 0
    
    return average_length



def pred_processor(pred):
    predic = []
    for i in range(len(pred)):
        predic.append(pred[i][0])

    predic = list(map(round, predic))
    final_pred = [i if i >= 1 else 1 for i in predic]
    final_pred = [i if i <= 6 else 6 for i in final_pred]
    return final_pred

def get_avg_w2v_vector(essay, model):
    vectors = [model.wv[word] for word in essay if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


cList = {
    "ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",  "could've": "could have",
    "couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not",
    "hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
    "he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is",
    "I'd": "I would","I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have","isn't": "is not",
    "it'd": "it had","it'd've": "it would have","it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam",
    "mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not",
    "mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
    "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would",
    "she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",
    "shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have",
    "that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is","they'd": "they would",
    "they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have",
    "to've": "to have","wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
    "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
    "what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is",
    "where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
    "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
    "wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would","y'all'd've": "you all would have",
    "y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have","you'll": "you will",
    "you'll've": "you will have","you're": "you are","you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    """
    Expand contractions in the given text based on the cList dictionary.
    
    Parameters:
    text (str): The input text containing contractions.
    c_re (re.Pattern): The compiled regex pattern for matching contractions.
    
    Returns:
    str: The text with contractions expanded.
    """
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def removeHTML(x):
    """
    Remove HTML tags from the given text.
    
    Parameters:
    x (str): The input text containing HTML tags.
    
    Returns:
    str: The text with HTML tags removed.
    """
    html = re.compile(r'<.*?>')
    return html.sub(r'', x)


def dataPreprocessing(x):
    """
    Preprocess the input text by performing a series of cleaning steps:
    - Convert to lowercase
    - Remove HTML tags
    - Remove mentions
    - Remove contractions
    - Remove digits
    - Remove URLs
    - Remove extra whitespaces
    - Expand contractions
    - Remove repeated punctuation
    
    Parameters:
    x (str): The input text to preprocess.
    
    Returns:
    str: The cleaned and preprocessed text.
    """
    x = x.lower()  # Convert to lowercase
    x = removeHTML(x)  # Remove HTML tags
    x = re.sub("@\w+", '', x)  # Remove mentions
    x = re.sub("'\d+", '', x)  # Remove contractions
    x = re.sub("\d+", '', x)  # Remove digits
    x = re.sub("http\w+", '', x)  # Remove URLs
    x = re.sub(r"\s+", " ", x)  # Remove extra whitespaces
    x = expandContractions(x)  # Expand contractions
    x = re.sub(r"\.+", ".", x)  # Remove repeated periods
    x = re.sub(r"\,+", ",", x)  # Remove repeated commas
    x = x.strip()  # Remove leading and trailing whitespaces
    x = clean_text(x)
    return x

def extract_features(text):
    words = word_tokenize(text)
    num_words = len(words)
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    spell_error = count_spelling_errors(text)
    sent_count = sentence_count(text)
    average_sent_length = average_sentence_length(text)
    para_count = count_paragraphs(text)
    average_para_length = average_paragraph_length(text)
    return num_words, avg_word_length, spell_error, sent_count, average_sent_length, para_count, average_para_length

