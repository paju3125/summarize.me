
import pandas as pd
import nltk


# Define a function to perform POS tagging on a given sentence
def pos_tag_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens


# Define a function to extract noun phrases from a given tagged sentence
def extract_noun_phrases(tagged_sentence):
    noun_phrase = ""
    noun_phrases = []
    for tagged_token in tagged_sentence:
        if tagged_token[1].startswith("NN"):
            noun_phrase += tagged_token[0] + " "
        elif noun_phrase != "":
            noun_phrases.append(noun_phrase.strip())
            noun_phrase = ""
    return noun_phrases


# Define a function to identify key verbs in a given tagged sentence
def identify_key_verbs(tagged_sentence):
    key_verbs = []
    for tagged_token in tagged_sentence:
        if tagged_token[1].startswith("VB"):
            key_verbs.append(tagged_token[0])
    return key_verbs



# Define a function to rank sentences based on number of noun phrases and key verbs
def rank_sentences(sentences):
    sentence_scores = []
    for sentence in sentences:
        tagged_sentence = pos_tag_sentence(sentence)
        score = 0
        for tagged_token in tagged_sentence:
            if tagged_token[1].startswith("NN"):
                score += 1
            elif tagged_token[1].startswith("VB"):
                score += 2
        sentence_scores.append(score)
    sorted_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]
    return sorted_sentences


# Generate summary for aur data.txt text
def POS_Summary(raw_text):
    sentences = nltk.sent_tokenize(raw_text)
    ranked_sentences = rank_sentences(sentences)
    summary = " ".join(ranked_sentences[:8])
    return summary
