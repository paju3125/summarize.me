from __future__ import unicode_literals
from flask import render_template
from flask import Flask
from flask import request

from flask import Flask,render_template,url_for,request, jsonify

from spacy_summarization import text_summarizer

from posTagging import POS_Summary
# from gensim.summarization.summarizer import summarize
# from nltk_summarization import nltk_summarizer


from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline
# Pick model
model_name = "google/pegasus-cnn_dailymail"

# Load pretrained tokenizer
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Define summarization pipeline 
summarizer = pipeline(
    "summarization", 
    model=model_name, 
    tokenizer=pegasus_tokenizer, 
    framework="pt"
)


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen

from nlp_advanced import article_summarize

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
# import time
import spacy
nlp = spacy.load("en_core_web_sm")

import time

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocessTxt():
    data = request.form.get('text')
    print(data)

    contractions_dict = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "doesn’t": "does not",
    "don't": "do not",
    "don’t": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y’all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "ain’t": "am not",
    "aren’t": "are not",
    "can’t": "cannot",
    "can’t’ve": "cannot have",
    "’cause": "because",
    "could’ve": "could have",
    "couldn’t": "could not",
    "couldn’t’ve": "could not have",
    "didn’t": "did not",
    "doesn’t": "does not",
    "don’t": "do not",
    "don’t": "do not",
    "hadn’t": "had not",
    "hadn’t’ve": "had not have",
    "hasn’t": "has not",
    "haven’t": "have not",
    "he’d": "he had",
    "he’d’ve": "he would have",
    "he’ll": "he will",
    "he’ll’ve": "he will have",
    "he’s": "he is",
    "how’d": "how did",
    "how’d’y": "how do you",
    "how’ll": "how will",
    "how’s": "how is",
    "i’d": "i would",
    "i’d’ve": "i would have",
    "i’ll": "i will",
    "i’ll’ve": "i will have",
    "i’m": "i am",
    "i’ve": "i have",
    "isn’t": "is not",
    "it’d": "it would",
    "it’d’ve": "it would have",
    "it’ll": "it will",
    "it’ll’ve": "it will have",
    "it’s": "it is",
    "let’s": "let us",
    "ma’am": "madam",
    "mayn’t": "may not",
    "might’ve": "might have",
    "mightn’t": "might not",
    "mightn’t’ve": "might not have",
    "must’ve": "must have",
    "mustn’t": "must not",
    "mustn’t’ve": "must not have",
    "needn’t": "need not",
    "needn’t’ve": "need not have",
    "o’clock": "of the clock",
    "oughtn’t": "ought not",
    "oughtn’t’ve": "ought not have",
    "shan’t": "shall not",
    "sha’n’t": "shall not",
    "shan’t’ve": "shall not have",
    "she’d": "she would",
    "she’d’ve": "she would have",
    "she’ll": "she will",
    "she’ll’ve": "she will have",
    "she’s": "she is",
    "should’ve": "should have",
    "shouldn’t": "should not",
    "shouldn’t’ve": "should not have",
    "so’ve": "so have",
    "so’s": "so is",
    "that’d": "that would",
    "that’d’ve": "that would have",
    "that’s": "that is",
    "there’d": "there would",
    "there’d’ve": "there would have",
    "there’s": "there is",
    "they’d": "they would",
    "they’d’ve": "they would have",
    "they’ll": "they will",
    "they’ll’ve": "they will have",
    "they’re": "they are",
    "they’ve": "they have",
    "to’ve": "to have",
    "wasn’t": "was not",
    "we’d": "we would",
    "we’d’ve": "we would have",
    "we’ll": "we will",
    "we’ll’ve": "we will have",
    "we’re": "we are",
    "we’ve": "we have",
    "weren’t": "were not",
    "what’ll": "what will",
    "what’ll’ve": "what will have",
    "what’re": "what are",
    "what’s": "what is",
    "what’ve": "what have",
    "when’s": "when is",
    "when’ve": "when have",
    "where’d": "where did",
    "where’s": "where is",
    "where’ve": "where have",
    "who’ll": "who will",
    "who’ll’ve": "who will have",
    "who’s": "who is",
    "who’ve": "who have",
    "why’s": "why is",
    "why’ve": "why have",
    "will’ve": "will have",
    "won’t": "will not",
    "won’t’ve": "will not have",
    "would’ve": "would have",
    "wouldn’t": "would not",
    "wouldn’t’ve": "would not have",
    "y’all": "you all",
    "y’all": "you all",
    "y’all’d": "you all would",
    "y’all’d’ve": "you all would have",
    "y’all’re": "you all are",
    "y’all’ve": "you all have",
    "you’d": "you would",
    "you’d’ve": "you would have",
    "you’ll": "you will",
    "you’ll’ve": "you will have",
    "you’re": "you are",
    "you’re": "you are",
    "you’ve": "you have",
    }
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    # Function expand the contractions if there's any
    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)

    # Function to clean the html from the article
    def cleanhtml(raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext

    # Text preprocessing function
    def preprocess_text(text):
        # Convert text to lowercase
        # text = text.lower()

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Removing the HTML
        # text = text.apply(lambda x: cleanhtml(x))
        text = cleanhtml(text)
        
        # Removing the email ids
        text =re.sub('\S+@\S+','', text)
        
        # Removing The URLS
        text = re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",'', text)

        # Removing the '\xa0'
        text = text.replace("\xa0", " ")
        
        # Removing the contractions
        text =  expand_contractions(text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Perform POS tagging
        tagged_tokens = pos_tag(tokens)
        

        # Join tokens back into a string
        preprocessed_text = " ".join(tokens)

        return preprocessed_text

        # # Preprocess example text
        # preprocessed_text = preprocess_text(example_text)

        # # Print preprocessed text
        # print(preprocessed_text)

    data = preprocess_text(data)
    time.sleep(1)
    return {'status':True, 'preprocessed_text': data}

@app.route('/generateSummary', methods=['GET', 'POST'])
def generateSummary():
    print('hiii')
    data = request.get_json()
    # print(data)
    print(type(data))
    preprocessed_text = data.get('preprocessed_text')
    print(preprocessed_text)
    # print(type(data))
    # preprocessed_text = data['response']['preprocessed_text']
    
    # preprocessed_text = request.form.get('preprocessed_text')
    # print(preprocessed_text)
    # Define PEGASUS model
    
    # model_name = data['additionalData']
    # print(model_name)
    
    # Create tokens
    tokens = pegasus_tokenizer(preprocessed_text, truncation=True, padding="longest", return_tensors="pt")
    # Summarize text
    encoded_summary = pegasus_model.generate(**tokens)

    # Decode summarized text
    decoded_summary = pegasus_tokenizer.decode(
        encoded_summary[0],
        skip_special_tokens=True
    )
    try:
        summary = summarizer(preprocessed_text, min_length=100, max_length=200)
        final_summary = summary[0]['summary_text']
    except:
        # Set the maximum token limit
        max_token_limit = 1024

        # Split the input text into smaller chunks
        chunks = [preprocessed_text[i:i+max_token_limit] for i in range(0, len(preprocessed_text), max_token_limit)]

        # Initialize an empty list to store the individual summaries
        summaries = []

        # Generate summary for each chunk
        for chunk in chunks:
            summary = summarizer(chunk, min_length=50, max_length=150)[0]['summary_text']
            summaries.append(summary)
        
        final_summary = ' '.join(summaries)

    final_summary = final_summary.replace('<n>', '')
    # time.sleep(1)
    return {'status':True, 'summary':final_summary}


@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	if request.method == 'POST':
		raw_url = request.form['url']
		rawtext = get_text(raw_url)
		
	return {'status':1, 'rawText':rawtext}

@app.route('/comparer',methods=['GET','POST'])
def comparer():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        rawtext.replace('\n', '')
        rawtext.replace('\t', '')
        # print(rawtext)
        final_reading_time = readingTime(rawtext)
        final_summary_spacy = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary_spacy)
        final_summary_posTagging = POS_Summary(rawtext)
        summary_reading_time_posTagging = readingTime(final_summary_posTagging)
        
        final_summary_adv_nlp = article_summarize(rawtext)
        summary_reading_time_adv_nlp = readingTime(final_summary_adv_nlp)
        
        print(final_summary_posTagging)
        # Gensim Summarizer
        # final_summary_gensim = summarize(rawtext)
        # summary_reading_time_gensim = readingTime(final_summary_gensim)
        # NLTK
        # final_summary_nltk = nltk_summarizer(rawtext)
        # summary_reading_time_nltk = readingTime(final_summary_nltk)
        # Sumy
        final_summary_sumy = sumy_summary(rawtext)
        summary_reading_time_sumy = readingTime(final_summary_sumy) 

        end = time.time()
        final_time = end-start
        return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy, final_summary_posTagging=final_summary_posTagging, summary_reading_time_posTagging=summary_reading_time_posTagging, summary_reading_time_adv_nlp=summary_reading_time_adv_nlp,final_summary_adv_nlp = final_summary_adv_nlp)
    return render_template('compare_template.html')

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,10)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

if __name__ == '__main__':
    app.run(debug=True)