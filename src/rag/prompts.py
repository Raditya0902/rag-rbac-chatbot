from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# The exact sentence the model must use when the context lacks the answer.
# The eval pipeline detects refusals by this text, so keep the two in sync.
REFUSAL_MESSAGE = "I don't have enough information to answer that."

# Appended when the context states the facts but not how they connect. Must not
# contain REFUSAL_MESSAGE, or the eval would count a partial answer as a refusal.
MISSING_LINK_MESSAGE = "The documents don't describe how these are connected."

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
            "Do not state purposes, causes, consequences or connections between facts unless the "
            "context states them; do not apply a policy to the user's situation beyond what the "
            "context says. When a question asks how things relate, what drove something or what it "
            "implies, and the context does not say, first give every related fact the context does "
            f"state, then end with: '{MISSING_LINK_MESSAGE}' Never open with that sentence, and "
            "never use it to skip facts the context gives. "
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
