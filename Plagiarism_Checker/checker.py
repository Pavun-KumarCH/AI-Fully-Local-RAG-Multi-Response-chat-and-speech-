import streamlit as st
import re
import nltk
import sys
import Plagiarism_Checker.webSearch as webSearch

from nltk.corpus import stopwords

# Function to split text into n-grams
def getQueries(text, n):
    sentenceEnders = re.compile("['.!?]")
    sentenceList = sentenceEnders.split(text)
    sentencesplits = []
    en_stops = set(stopwords.words('english'))

    for sentence in sentenceList:
        x = re.compile(r'\W+', re.UNICODE).split(sentence)
        for word in x:
            if word.lower() in en_stops:
                x.remove(word)
        x = [ele for ele in x if ele != '']
        sentencesplits.append(x)

    finalq = []
    for sentence in sentencesplits:
        l = len(sentence)
        if l > n:
            l = int(l/n)
            index = 0
            for i in range(0, l):
                finalq.append(sentence[index:index+n])
                index = index + n-1
                if index+n > l:
                    index = l-n-1
            if index != len(sentence):
                finalq.append(sentence[len(sentence)-index:len(sentence)])
        else:
            if l > 4:
                finalq.append(sentence)

    return finalq

# Function to check plagiarism using web search and cosine similarity
def plagiarism_check(input_text):
    n = 20
    queries = getQueries(input_text, n)

    query_sentences = [' '.join(sentence) for sentence in queries]
    output = {}
    cosine_similarities = {}
    num_queries = len(query_sentences)

    if num_queries > 100:
        num_queries = 100  # Limit to first 100 queries for API efficiency

    for i, query in enumerate(query_sentences[:num_queries]):
        if query.strip() == "":
            continue  # Skip empty queries

        output, cosine_similarities, errorCount = webSearch.searchWeb(query, output, cosine_similarities)
        if errorCount:
            st.write(f"Error occurred while searching for: {query}")

        sys.stdout.flush()

    total_plagiarism = 0
    matching_sources = {}

    for link in output:
        percentage = (output[link] * cosine_similarities[link] * 100) / num_queries
        if percentage > 10:
            total_plagiarism += percentage
            matching_sources[link] = percentage
        elif cosine_similarities[link] == 1:
            total_plagiarism += percentage

    return total_plagiarism, matching_sources
