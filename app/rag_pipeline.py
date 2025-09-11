# app/rag_pipeline.py
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from app.helpers import load_model, load_memory, format_sources


# import your retrievers/reranker setup here...

def build_rag_pipeline():
    model = load_model()
    summary_memory = load_memory(model)

    return model, summary_memory


def answer_question(question: str, retriever, model, memory):
    def build_inputs_with_memory(inputs):
        docs = inputs["retrieved"]
        return {
            "history": memory.load_memory_variables({}).get("history", ""),  # conversation summary
            "context": "\n\n".join(doc.page_content for doc in docs),
            "question": inputs["question"],
            "sources": [doc.metadata for doc in docs]
        }

    prompt = PromptTemplate(
        template="""
                 You are a helpful assistant.
                 Use the conversation history and the provided transcript context to answer.
                 If the context is insufficient, just say you don't know.

                 Conversation History: {history}
                 Context:{context}
                 Question: {question}
               """,
        input_variables=['history', 'context', 'question']
    )
    parser = StrOutputParser()

    llm_chain = prompt | model | parser

    parallel_chain = RunnableParallel({
        "retrieved": retriever,
        "question": RunnablePassthrough()
    })

    final_chain = RunnableParallel({
        "answer": parallel_chain | RunnableLambda(build_inputs_with_memory) | llm_chain,
        "sources": retriever | RunnableLambda(format_sources)
    })

    result = final_chain.invoke(question)
    memory.save_context({"input": question}, {"output": result["answer"]})

    return result
