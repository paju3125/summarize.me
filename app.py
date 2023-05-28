from __future__ import unicode_literals
from flask import render_template
from flask import Flask
from flask import request

from flask import Flask,render_template,url_for,request

from spacy_summarization import text_summarizer

from posTagging import POS_Summary
# from gensim.summarization.summarizer import summarize
# from nltk_summarization import nltk_summarizer

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
    time.sleep(1)
    return {'status':True}

@app.route('/generateSummary', methods=['GET', 'POST'])
def generateSummary():
    data = request.form.get('status')
    print(data)
    time.sleep(1)
    return {'status':True}


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
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

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
	summary = lex_summarizer(parser.document,3)
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