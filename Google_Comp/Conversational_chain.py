import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain.vectorstores import FAISS  

# Initialize memory history
memory_history = []

def get_conversational_chain(selected_prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", 
                                   temperature = 0.7, # Lower for precision
                                   top_p=0.9 # Allow some creativity while remaining focused
                                   ) 
    prompt = PromptTemplate(template=selected_prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, selected_prompt_template):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
   
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(selected_prompt_template)

    # Create a string representation of the memory history
    context = "\n".join(memory_history)  # Combine all previous interactions into a single context

    response = chain(
        {"input_documents": docs, "question": user_question, "context": context},
        return_only_outputs=True
    )
    
    print(response)
    st.markdown("**Agent :**")
    response_text = response.get("output_text", "No output generated.")
    st.write(response_text)

    # Update memory history with user question and agent response
    memory_history.append(f"User: {user_question}")
    memory_history.append(f"Agent: {response_text}")

# Streamlit app setup and main loop would go here...
