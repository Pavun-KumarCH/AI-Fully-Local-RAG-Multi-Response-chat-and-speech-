# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 07:13:09 2024

@author: RONY
"""

import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load the environment variables (e.g., API key)
load_dotenv()

# Initialize the GoogleGenerativeAI model
api_key = os.getenv('GOOGLE_GENAI_API_KEY')
model = GoogleGenerativeAI(api_key=api_key, model="gemini-1.0-pro")


def react_prompting(prompt, num_iterations=3):
    # Initial prompt to generate questions
    initial_prompt = f"""
    You are an AI assistant specialized in creating educational content. 
    Your task is to generate a set of multiple-choice questions based on the following topic: "{prompt}". 
    Focus strictly on the topic and ensure all questions and answers are relevant and accurate.

    Format your output as follows:
    - Question: [Your question here]
    - A. [Answer choice A]
    - B. [Answer choice B]
    - C. [Answer choice C]
    - D. [Answer choice D]
    - Explanation: [Why the correct answer is correct, and why the other options are incorrect]

    Ensure that each question tests a key concept from the topic and that all answer choices are plausible and relevant to the topic.
    """

    # Generate initial set of questions
    response = model.generate(prompts=[initial_prompt])
    questions = response.generations[0][0].text
    
    for _ in range(num_iterations):
        # Process the generated questions and prepare feedback
        feedback = prepare_feedback(questions, prompt)

        # Refine prompt based on feedback
        refined_prompt = f"""
        Based on the previous set of questions, please refine and improve the questions.
        Incorporate the following feedback: {feedback}
        Ensure that all questions remain strictly relevant to the topic: "{prompt}".

        Continue to format your output as follows:
        - Question: [Your question here]
        - A. [Answer choice A]
        - B. [Answer choice B]
        - C. [Answer choice C]
        - D. [Answer choice D]
        - Explanation: [Why the correct answer is correct, and why the other options are incorrect]
        """

        # Generate refined set of questions
        response = model.generate(prompts=[refined_prompt])
        questions = response.generations[0][0].text

    return questions

def prepare_feedback(questions, topic):
    # Dummy function to prepare feedback for the next iteration
    # This function should be customized based on actual evaluation of the questions
    # Here we ensure feedback is targeted to keep the questions relevant to the topic
    feedback = "Ensure all questions are highly relevant to the topic: " + topic
    return feedback

# Example usage
main_topic = "Organic Chemistry"
refined_questions = react_prompting(main_topic, num_iterations=3)

# Print the refined questions with React Prompting
print(refined_questions)