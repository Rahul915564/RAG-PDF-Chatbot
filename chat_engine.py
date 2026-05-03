import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_PROMPT_ENGLISH = """You are a helpful, knowledgeable AI assistant that answers questions based on the provided document context.

Instructions:
- Answer questions accurately based ONLY on the provided context/documents.
- If the answer is not in the context, say clearly: "I don't find information about this in the uploaded documents."
- ALWAYS respond in English regardless of the language the user writes in.
- Be concise but thorough. Use bullet points for lists when appropriate.
- Always be polite and helpful.

Context from documents:
{context}

Chat History:
{chat_history}"""

SYSTEM_PROMPT_HINDI = """आप एक सहायक और जानकार AI सहायक हैं जो दिए गए दस्तावेज़ संदर्भ के आधार पर प्रश्नों का उत्तर देते हैं।

निर्देश:
- केवल दिए गए संदर्भ/दस्तावेज़ों के आधार पर सटीक उत्तर दें।
- यदि संदर्भ में उत्तर नहीं है, तो स्पष्ट रूप से कहें: "अपलोड किए गए दस्तावेज़ों में इस विषय में जानकारी नहीं मिली।"
- उपयोगकर्ता चाहे किसी भी भाषा में पूछे, हमेशा हिंदी में उत्तर दें।
- संक्षिप्त लेकिन पूर्ण उत्तर दें। सूचियों के लिए बुलेट पॉइंट का उपयोग करें।
- हमेशा विनम्र और सहायक रहें।

दस्तावेज़ों से संदर्भ:
{context}

चैट इतिहास:
{chat_history}"""


def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        max_tokens=2048,
    )


def get_answer(
    question: str,
    vector_store,
    chat_history: List[Dict],
    language: str = "English",
) -> Dict[str, Any]:
    llm = get_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    history_text = ""
    for msg in chat_history[-6:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    system_template = SYSTEM_PROMPT_HINDI if language == "Hindi" else SYSTEM_PROMPT_ENGLISH

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "chat_history": history_text,
        "question": question,
    })

    answer = response.content

    sources = []
    seen = set()
    for doc in relevant_docs:
        meta = doc.metadata
        source_file = meta.get("source_file", meta.get("source", "Unknown"))
        page = meta.get("page", 0) + 1
        snippet = doc.page_content[:200].strip().replace("\n", " ")
        key = (source_file, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": source_file,
                "page": page,
                "snippet": snippet + ("..." if len(doc.page_content) > 200 else ""),
            })

    return {"answer": answer, "sources": sources}
