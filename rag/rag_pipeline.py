from rag.retriever import get_relevant_chunks
from rag.llm_interface import query_llm
from pathlib import Path

TEMPLATE_PATH = Path("llm/prompt_template.txt")
DEFAULT_TEMPLATE = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say so.

Context:
{context}

Question:
{question}
"""

def load_prompt_template():
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text()
    return DEFAULT_TEMPLATE

def build_prompt(chunks, question):
    """
    Build a prompt by combining each chunk with its source.
    Each chunk is annotated with the source from which it came.
    """
    context_lines = []
    for chunk in chunks:
        # You can format the source info as you wish.
        context_lines.append(f"[{chunk['source']}]:\n{chunk['text']}")
    context = "\n\n".join(context_lines)
    template = load_prompt_template()
    return template.format(context=context, question=question)

def run_rag(question):
    print("🔍 Retrieving relevant chunks...")
    chunks = get_relevant_chunks(question)

    print("\n✅ Retrieved Chunks (showing sources):")
    for idx, chunk in enumerate(chunks, start=1):
        preview = chunk['text'][:100].replace("\n", " ")  # short preview
        print(f"   {idx}. [{chunk['source']}]: {preview}...")

    print("\n🧾 Building prompt with context...")
    prompt = build_prompt(chunks, question)

    print("🧠 Querying local LLM...")
    answer = query_llm(prompt)

    return answer, chunks

