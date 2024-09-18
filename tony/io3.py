# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 00:48:24 2024

@author: Rony Joseph
"""

import time
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Configure the API key securely

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path = r"C:\Users\RONY\venv1\myproj.env")



# Initialize the GoogleGenerativeAI model
llm = GoogleGenerativeAI(model="models/gemini-1.0-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
                         
def google_search(query):
    """Perform a Google search and return the first few results."""
    search_url = "https://www.google.com/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": query}
    response = requests.get(search_url, headers=headers, params=params)
    soup = BeautifulSoup(response.text, "html.parser")
    results = [a['href'] for a in soup.find_all('a', href=True) if 'url?q=' in a['href']]
    return results[:5]  # Return the top 5 results

def check_plagiarism(text, threshold=0.7):
    """Check if the given text is plagiarized by comparing it with online content."""
    sentences = text.split('.')
    plagiarism_results = []
    for sentence in sentences:
        query = sentence.strip()
        if query:
            results = google_search(query)
            for result in results:
                try:
                    result_content = requests.get(result).text
                except requests.RequestException:
                    continue
                
                documents = [query, result_content]
                tfidf_vectorizer = TfidfVectorizer().fit_transform(documents)
                cosine_similarities = cosine_similarity(tfidf_vectorizer[0:1], tfidf_vectorizer[1:]).flatten()
                similarity_score = cosine_similarities[0]

                if similarity_score > threshold:
                    plagiarism_results.append((query, result, similarity_score))
    return plagiarism_results

def generate_io_prompt(prompt, io_structure, iterations=1, retries=3):
    """Generates structured prompts with retry and timeout handling."""
    for _ in range(iterations):
        full_prompt = f"{io_structure}\n\n{prompt}"
        template = "{full_prompt}"

        prompt_template = PromptTemplate.from_template(template)
        chain = prompt_template | llm

        for attempt in range(retries):
            try:
                response = chain.invoke(input={"full_prompt": full_prompt})

                if isinstance(response, dict) and "text" in response:
                    generated_text = response["text"]
                elif isinstance(response, str):
                    generated_text = response
                else:
                    raise ValueError("Unexpected response format")

                prompt = generated_text
                break

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    return generated_text

def generate_correct_answers(sub_topic, questions_text):
    """Generate correct answers for the given questions on a sub-topic."""
    prompt = f"""
    For the sub-topic '{sub_topic}', provide correct answers for the following questions:
    
    {questions_text}
    
    Output: Provide the correct answers clearly listed.
    """
    io_structure = "Input: Provide correct answers for the given questions. Output: List of correct answers."
    return generate_io_prompt(prompt, io_structure)

def generate_sub_topics(main_topic, io_structure):
    prompt = f"Break down the main topic '{main_topic}' into 20 important sub-topics."
    return generate_io_prompt(prompt, io_structure)

# Streamlit app layout
st.title("Question Generator")

# User input for the main topic
main_topic = st.text_input("Enter the Main Topic:")

# Define IO structures
sub_topic_io_structure = """
Input: Break down the given main topic into 20 important sub-topics.
Output: Provide a list of 20 relevant and significant sub-topics related to the main topic.
"""

question_io_structure = """
Input: Provide a prompt related to a specific sub-topic.
Output: Generate Five Multiple Choice Questions (MCQs), Five Short Answer Questions, and Five Long Answer Questions on the Provided sub-topic.

Guidelines:
1. Clearly list the input sub-topic.
2. Generate a structured response with separate sections for MCQs, Short Answer Questions, and Long Answer Questions.
3. Ensure each question is well-formulated and relevant to the sub-topic and Plagiarism-Free.
"""

if st.button("Generate Questions and Answers"):
    if main_topic:
        st.write(f"Generating Sub-Topics for the Main Topic '{main_topic}'...\n")
        try:
            sub_topics_text = generate_sub_topics(main_topic, sub_topic_io_structure)
            st.write(f"Sub-Topics generated:\n{sub_topics_text}\n")

            # Convert generated text into a list of sub-topics
            sub_topics = [line.strip() for line in sub_topics_text.split('\n') if line.strip()]

            # Generate questions and correct answers for each sub-topic
            for sub_topic in sub_topics:
                st.write(f"Generating questions for the Sub-Topic '{sub_topic}'...\n")
                initial_prompt = f"I want you to act as a teaching assistant to students at a university. Learn the required concepts and prepare a draft of: 1. Five Multiple Choice Questions 2. Five Short Answer Type Questions 3. Five Long Answer Type Questions on the topic of {sub_topic}."
                try:
                    questions_text = generate_io_prompt(initial_prompt, question_io_structure, retries=3)
                    st.write(f"Generated questions for the Sub-Topic '{sub_topic}':\n{questions_text}\n")

                    # Generate correct answers for the questions
                    st.write("Generating correct answers...\n")
                    answers_text = generate_correct_answers(sub_topic, questions_text)
                    st.write(f"Correct answers for the Sub-Topic '{sub_topic}':\n{answers_text}\n")

                    # Check the generated output for plagiarism
                    st.write("Checking for plagiarism...\n")
                    plagiarism_results = check_plagiarism(questions_text)
                    if plagiarism_results:
                        for query, result, score in plagiarism_results:
                            st.write(f"Plagiarism detected in the sentence: '{query}'")
                            st.write(f"Similarity score: {score:.2f}")
                            st.write(f"Matched with: {result}\n")
                    else:
                        st.write("No plagiarism detected.")

                except Exception as e:
                    st.error(f"Failed to generate questions or answers for the sub-topic '{sub_topic}': {e}")

        except Exception as e:
            st.error(f"Failed to generate Sub-Topics: {e}")
    else:
        st.warning("Please enter a Main Topic.")

