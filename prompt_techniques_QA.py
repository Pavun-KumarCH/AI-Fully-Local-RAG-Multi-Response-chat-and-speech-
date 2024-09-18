from langchain.prompts import PromptTemplate

# Default prompt for generating multiple perspectives on a user question
default_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

# Structured multiple-choice question generation
mcq_generation_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant specialized in creating educational content.
    Please generate questions multiple-choice questions based on the topic user requested".
    
    Format your output as follows:
    - Question: [Your {question} here]
    - A. [Answer choice A]
    - B. [Answer choice B]
    - C. [Answer choice C]
    - D. [Answer choice D]
    - Explanation: [Why the correct answer is correct, and why the other options are incorrect]

    Ensure that each question tests a key concept from the topic and that all answer choices are plausible.
    """
)

# Chain-of-Thought prompting for generating MCQs with detailed reasoning
chain_of_thought_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""You are an intelligent Quiz master, and the Uniqueness of Questions you generate is crucial to the conceptual understanding of future students!
    Please generate questions on high-quality multiple-choice questions on the topic given {question}.

    For each question:
    1. First, think of a key concept or idea from the topic.
    2. Then, generate a clear and concise question based on that concept.
    3. Next, provide 4 possible answer choices, with one being the correct answer and the other three being plausible distractors.
    4. Finally, explain why the correct answer is the right one.

    Use a step-by-step reasoning approach to ensure the questions are challenging but clear.
    """
)

# Simple prompt for multiple-choice question generation
simple_mcq_generation_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant with expertise in educational content creation. Your task is to generate {questions} high-quality multiple-choice questions to assess students' understanding of the topic: "{question}".

    Each question should:
    - Be clear, concise, and focused on a key concept or idea within the topic.
    - Challenge the student's knowledge while ensuring the question remains accessible.
    - Contain four possible answers:
      - One correct answer that directly addresses the question.
      - Three plausible distractors that could seem correct but are subtly different to test the student's comprehension.

    Additionally, ensure that:
    - Each question is conceptually distinct to cover a broad range of subtopics.
    - The questions progress in difficulty from easy to more challenging, helping learners reinforce their understanding.
    """
)

# Chain-of-Thought prompting with self-consistency for generating MCQs
cot_self_consistency_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI expert in educational content creation.
    Please generate high-quality multiple-choice questions on the topic given {question}".

    For each question:
    1. First, identify a key concept or idea from the topic.
    2. Generate a clear and concise question that tests that concept.
    3. Provide 4 answer choices: one correct answer and three plausible distractors.
    4. Explain why the correct answer is right and why the other options are incorrect.

    Use step-by-step reasoning to ensure the questions are challenging but clear.
    """
)


# ReACT Prompting with iterations for generating multiple-choice questions
react_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant specialized in creating educational content. 
    Your task is to generate a set of multiple-choice questions based on the following topic: "{question}". 
    Focus strictly on the topic and ensure all questions and answers are relevant and accurate.

    Format your output as follows:
    - Question: [Your question here]
    - A. [Answer choice A]
    - B. [Answer choice B]
    - C. [Answer choice C]
    - D. [Answer choice D]
    - Explanation: [Why the correct answer is correct, and why the other options are incorrect]

    Ensure that each question tests a key concept from the topic and that all answer choices are plausible and relevant to the topic.

    This process will be repeated on user requeseted times to refine and ensure a high-quality set of questions.
    """
)

from langchain.prompts import PromptTemplate

# Refined prompting template with feedback incorporation
refined_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""Based on the previous set of questions, please refine and improve the questions.
    Incorporate the following feedback by yourself.
    Ensure that all questions remain strictly relevant to the topic: "{question}".

    Continue to format your output as follows:
    - Question: [Your question here]
    - A. [Answer choice A]
    - B. [Answer choice B]
    - C. [Answer choice C]
    - D. [Answer choice D]
    - Explanation: [Why the correct answer is correct, and why the other options are incorrect]
    """
)
