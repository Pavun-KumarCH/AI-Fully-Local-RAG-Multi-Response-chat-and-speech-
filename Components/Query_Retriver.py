from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Modified query_model function with additional arguments
def query_model(vector_db, question, selected_prompt_template, selected_model):

    model = ChatOllama(model=selected_model, temperature = 0.4, top_p = 0.9)  # Use default ChatOllama for other models
    # Use the selected prompt template
    QUERY_PROMPT = selected_prompt_template
    
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(),
        llm=model,  # Use the correct model
        prompt=QUERY_PROMPT
    )

    # Generate the final response based on the retrieved context
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model  # Use the correct model
        | StrOutputParser()
    )

    response = chain.invoke(question)
    return response