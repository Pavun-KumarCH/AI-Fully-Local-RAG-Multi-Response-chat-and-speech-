# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 05:44:31 2024

@author: RONY
"""

import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from collections import Counter

# Load the environment variables (e.g., API key)
load_dotenv()

# Initialize the GoogleGenerativeAI model
api_key = os.getenv('GOOGLE_GENAI_API_KEY')
model = GoogleGenerativeAI(api_key=api_key, model="gemini-1.0-pro")

def generate_questions_with_IO_and_SC(prompt, num_questions=5, num_trials=3):
    # Input-Output prompt to guide the AI in generating questions
    question_prompt = f"""
    You are an AI assistant specialized in creating educational content. 
    Please generate {num_questions} multiple-choice questions based on the following topic: "{prompt}".
    
    Format your output as follows:
    - Question: [Your question here]
    - A. [Answer choice A]
    - B. [Answer choice B]
    - C. [Answer choice C]
    - D. [Answer choice D]
    - Explanation: [Why the correct answer is correct, and why the other options are incorrect]

    Ensure that each question tests a key concept from the topic and that all answer choices are plausible.
    """

    # Generate multiple responses to ensure self-consistency
    all_responses = []
    for _ in range(num_trials):
        # Using the generate() method and passing the prompt as a list
        response = model.generate(prompts=[question_prompt])
        
        # Extract the generated questions from the response
        all_responses.append(response.generations[0][0].text)

    # Process and aggregate the results using self-consistency
    consistent_questions = aggregate_consistent_answers(all_responses)
    
    return consistent_questions

def aggregate_consistent_answers(all_responses):
    # Count the frequency of each response to find the most consistent one
    questions_counter = Counter(all_responses)
    
    # Return the most frequent questions (top consistent results)
    most_consistent = questions_counter.most_common(1)[0][0]
    
    return most_consistent

# Example usage
main_topic = "Quantum Mechanics"
questions_with_IO_and_SC = generate_questions_with_IO_and_SC(main_topic, num_questions=5, num_trials=3)

# Print the generated questions with Input-Output prompting and self-consistency
print(questions_with_IO_and_SC)
