from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# The exact sentence the model must use when the context lacks the answer.
# The eval pipeline detects refusals by this text, so keep the two in sync.
REFUSAL_MESSAGE = "I don't have enough information to answer that."

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
            f"If the context lacks the answer, say exactly: '{REFUSAL_MESSAGE}' "
            "Never reveal information from departments the user has not been granted access to. "
            "Be concise and factual — no preamble. Answer in complete sentences that name "
            "what is being answered (e.g. 'Employees get N days of X leave per year.', "
            "not 'N days.'). For broad questions, cover every aspect the context addresses.\n\n"
            "Context:\n{context}"
        ),
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
