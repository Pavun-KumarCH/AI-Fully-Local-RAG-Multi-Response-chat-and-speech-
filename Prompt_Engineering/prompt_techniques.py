prompt_template = """You are an AI language model assistant. Your task is to generate five
                      distinct versions of the user's question to enhance the retrieval of relevant documents from
                      a vector database. By presenting multiple angles on the user question, you aim to assist the user in overcoming limitations associated with distance-based similarity searches. Ensure that these alternative questions maintain originality and are phrased in an active voice. For each version, you might consider rephrasing, specifying details, or exploring related concepts. 

                      Provide these alternative questions separated by newlines.
                      Original 

                  Context: \n{context}\n
                  Question: \n{question}\n

                  Answer:
                  """

prompt_template_1 = """Construct a graph where nodes represent key ideas, and edges illustrate the relationships among these ideas. 
                      Explain how these connections can lead to a coherent understanding of the content. 

                      Example: If the central idea is "Photosynthesis," related nodes might include "Chlorophyll," "Sunlight," and "Carbon Dioxide," with edges showing how each element interacts in the process.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Graph of Thought:
                      """

prompt_template_2 = """Create a tree structure where each branch represents an expansion of a core concept. 
                      Discuss how this hierarchy aids in comprehending the overall content. 

                      Example: Starting from the core concept "Healthy Living," branches might include "Nutrition," "Exercise," and "Mental Health," each with sub-branches detailing specific practices.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Tree of Thought:
                      """

prompt_template_3 = """Construct a graph to verify the relationships between key ideas. 
                      Discuss how this verification enhances your understanding of the content. 

                      Example: If exploring the concept of "Climate Change," verify relationships between "Greenhouse Gases," "Global Temperature," and "Weather Patterns."

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Graph of Verification:
                      """

prompt_template_4 = """Break down the summarization process into logical steps. 
                      Outline the intermediate steps necessary for generating a concise summary. 

                      Example: Steps might include identifying key points, condensing information, and rewriting in simpler terms.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Chain of Thought:
                      """

prompt_template_5 = """Explore all possible interpretations of this text. 
                      Describe how integrating multiple perspectives can yield a comprehensive understanding. 

                      Example: For a literary text, consider themes, character motivations, and author intent to form a holistic view.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      XOT of Thought:
                      """

prompt_template_6 = """Leverage domain-specific knowledge to enrich the chain-of-thought process. 
                      Explain how expert insights shape text interpretation and summarization. 

                      Example: In a medical context, a doctor might highlight nuances in terminology that could affect understanding.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      KD-CoT:
                      """

prompt_template_7 = """Maintain consistency throughout the chain-of-thought process in summarization. 
                      Discuss how to uphold internal consistency in the logical flow. 

                      Example: Ensure that all parts of the summary reference the same sources and do not contradict each other.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      COT-SC:
                      """

prompt_template_8 = """Generate questions that the text answers directly. 
                      Identify questions arising from the content and explain how answering them clarifies the material. 

                      Example: From a text about "Machine Learning," questions could include "What is overfitting?" and "How can it be avoided?"

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Self-Ask:
                      """

prompt_template_9 = """Critically evaluate the generated summary. 
                      Identify potential inaccuracies or misleading aspects, and propose corrections. 

                      Example: If a summary states that "All plants require sunlight," note that some plants thrive in low light and suggest revising.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Self-Critique:
                      """

prompt_template_10 = """Refine the summary iteratively by incorporating feedback. 
                       Discuss improvements made in each iteration to enhance clarity and accuracy. 

                       Example: After each review, address vague language, clarify technical terms, and ensure logical progression.

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Self-Refine:
                       """


prompt_template_11 = """Continuously improve the generated content. 
                       How can the content be polished to better align with the original text's intent?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Self-Refinement:
                       """

prompt_template_12 = """Engage in multiple rounds of prompting to gradually refine the output. 
                       How does each iteration improve the quality of the summary?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Iterative Prompting:
                       """

prompt_template_13 = """Use analogies to relate the content to familiar concepts. 
                       How can drawing parallels help in understanding complex ideas?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Analogical Prompting:
                       """

prompt_template_14 = """For each input (text), specify the expected output (summary). 
                       How can clear examples of input-output pairs guide the generation process?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Input-Output Prompting:
                       """

prompt_template_15 = """Start with the simplest possible summary and progressively add more details. 
                       How does gradually increasing complexity enhance the summary?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Least-to-Most Prompting:
                       """

prompt_template_16 = """Develop a plan to tackle the summarization problem and then execute it. 
                       What steps need to be followed to achieve the desired outcome?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Plan-and-Solve Prompting:
                       """

prompt_template_17 = """Break down the summarization into sequential tasks. 
                       How can each step logically follow the previous one to build a coherent summary?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Sequential Prompting:
                       """

prompt_template_18 = """Take a step back and reassess the current summary. 
                       What broader perspective can be gained by reevaluating the previous steps?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Step-Back Prompting:
                       """

prompt_template_19 = """Leverage memory of previous prompts to inform the current task. 
                       How can recalling past interactions help improve the summary?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       MemPrompt:
                       """

prompt_template_20 = """Focus on creating a dense, information-rich summary. 
                       How can the most important details be packed into a concise form?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Chain of Density:
                       """

prompt_template_21 = """Generate a summary in a structured format, such as JSON, that can be easily reversed into the original content. 
                       How does this structure aid in understanding?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Reverse JSON Prompting:
                       """

prompt_template_22 = """Apply symbolic logic to the content. 
                       How can formal reasoning techniques help clarify the underlying arguments?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Symbolic Reasoning:
                       """

prompt_template_23 = """Use the text to generate new knowledge or insights. 
                       What novel ideas can be derived from the content?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Generated Knowledge:
                       """

prompt_template_24 = """Use programmatic aids to enhance the summarization process. 
                       How can algorithms assist in generating more accurate summaries?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       PAL:
                       """

prompt_template_25 = """Ensure that multiple summaries of the same content are consistent with each other. 
                       How can consistency across different iterations be achieved?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       Meta-Ask Prompting:
                       """

prompt_template_26 = """React to the generated content by providing immediate feedback. 
                       How can interactive responses guide the improvement of the summary?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       ReAct:
                       """

prompt_template_27 = """Utilize automatic reasoning tools to enhance the summary. 
                       What tools can assist in analyzing and condensing the content?

                       Context: \n{context}\n
                       Question: \n{question}\n

                       ART:
                       """

prompt_template_28 = """Provide a few examples of how to analyze and summarize the content. 
                        Based on these examples, analyze and summarize the following content.

                        Context:\n{context}\n
                        Question: \n{question}\n

                        Response:
                        """

prompt_template_29 = """Analyze the content from the provided context and generate a summary without any specific examples.

                      Context:\n{context}\n
                      Question: \n{question}\n
                      
                      Summary:
                      """

prompt_template_30 = """Break down the process of summarizing this content into a series of logical steps. 
                      Consider the structure of the content and the key points to cover.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Chain of Thought:
                      """

prompt_template_31 = """Follow these explicit instructions to summarize the content:
                      1. Identify the main sections.
                      2. Extract key information from each section.
                      3. Combine the extracted information into a coherent summary.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Summary:
                      """

prompt_template_32 = """Respond as if you are a professional analyst with expertise in summarizing complex documents. 
                      Adopt their style and perspective to summarize the provided content.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Summary:
                      """

prompt_template_33 = """Use the context provided to generate a detailed summary. 
                      Ensure that the summary reflects the key themes and information presented.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Summary:
                      """

prompt_template_34 = """Ensure the generated summary captures the nuances of the text. 
                      Consider any implicit meanings or underlying themes that should be included.

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Summary:
                      """
prompt_template_35 = """Incorporate any previous feedback or notes into the current summary. 
                      How can past feedback enhance the accuracy of the current summary?

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Feedback-Integrated Summary:
                      """

prompt_template_36 = """Compare the generated summary with a reference summary. 
                      What differences exist, and how can they be reconciled?

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Comparative Analysis:
                      """

prompt_template_37 = """Develop a summary that anticipates potential questions or objections from the reader. 
                      How can the summary preemptively address these points?

                      Context: \n{context}\n
                      Question: \n{question}\n

                      Anticipatory Summary:
                      """
