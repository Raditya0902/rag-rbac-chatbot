from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Rewrites a follow-up question into a standalone question so retrieval works
# correctly even when the user refers to something from earlier in the chat.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the follow-up question as a self-contained standalone question "
        "using the conversation history for context. "
        "If no rewrite is needed, return the question unchanged. "
        "Output ONLY the rewritten question, nothing else.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful internal company assistant. "
            "Answer the user's question directly and specifically — begin your response by addressing exactly what was asked. "
            "Use ONLY facts explicitly stated in the context below. Do not add, infer, or speculate. "
            "If the context lacks the answer, say exactly: 'I don't have enough information to answer that.' "
            "Never reveal information from departments the user has not been granted access to. "
            "Be concise and factual — avoid preamble or restating the question.\n\n"
            "Context:\n{context}"
        ),
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
