# Default prompt for generating multiple perspectives on a user question
default_prompt = """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.

Context: 
{context}

Question: 
{question}

Answer:
"""

# Structured multiple-choice question generation
mcq_generation_prompt = """You are an AI assistant specialized in creating educational content. Your task is to generate {num_questions} multiple-choice questions based on the following topic: "{topic}". Ensure that each question is clear, concise, and focused on a key concept or idea within the topic.

Format your output as follows:
- Question: [Your question here]
- A. [Answer choice A]
- B. [Answer choice B]
- C. [Answer choice C]
- D. [Answer choice D]
- Explanation: [Why the correct answer is correct, and why the other options are incorrect]

Context: 
{context}

Answer:
"""

# Chain-of-Thought prompting for generating MCQs with detailed reasoning
chain_of_thought_prompt = """You are an intelligent Quiz master, and the uniqueness of questions you generate is crucial to the conceptual understanding of future students! Please generate {num_questions} high-quality multiple-choice questions on the topic of "{topic}". 

For each question:
1. First, think of a key concept or idea from the topic.
2. Then, generate a clear and concise question based on that concept.
3. Next, provide 4 possible answer choices, with one being the correct answer and the other three being plausible distractors.
4. Finally, explain why the correct answer is the right one.

Context: 
{context}

Answer:
"""

# Simple prompt for multiple-choice question generation
simple_mcq_generation_prompt = """You are an AI assistant with expertise in educational content creation. Your task is to generate {num_questions} high-quality multiple-choice questions to assess students' understanding of the topic: "{topic}". 

Each question should:
- Be clear, concise, and focused on a key concept or idea within the topic.
- Challenge the student's knowledge while ensuring the question remains accessible.
- Contain four possible answers:
  - One correct answer that directly addresses the question.
  - Three plausible distractors that could seem correct but are subtly different to test the student's comprehension.

Context: 
{context}

Answer:
"""

# Chain-of-Thought prompting with self-consistency for generating MCQs
cot_self_consistency_prompt = """You are an AI expert in educational content creation. Please generate {num_questions} high-quality multiple-choice questions on the topic of "{topic}". 

For each question:
1. First, identify a key concept or idea from the topic.
2. Generate a clear and concise question that tests that concept.
3. Provide 4 answer choices: one correct answer and three plausible distractors.
4. Explain why the correct answer is right and why the other options are incorrect.

Context: 
{context}

Answer:
"""

# ReACT Prompting with iterations for generating multiple-choice questions
react_prompt = """You are an AI assistant specialized in creating educational content. Your task is to generate a set of multiple-choice questions based on the following topic: "{topic}". Focus strictly on the topic and ensure all questions and answers are relevant and accurate.

Format your output as follows:
- Question: [Your question here]
- A. [Answer choice A]
- B. [Answer choice B]
- C. [Answer choice C]
- D. [Answer choice D]
- Explanation: [Why the correct answer is correct, and why the other options are incorrect]

Context: 
{context}

Answer:
"""

# Refined prompt with feedback incorporation
refined_prompt = """Based on the previous set of questions, please refine and improve the questions. Incorporate the following feedback: {feedback}. Ensure that all questions remain strictly relevant to the topic: "{topic}".

Continue to format your output as follows:
- Question: [Your question here]
- A. [Answer choice A]
- B. [Answer choice B]
- C. [Answer choice C]
- D. [Answer choice D]
- Explanation: [Why the correct answer is correct, and why the other options are incorrect]

Context: 
{context}

Answer:
"""
