MAX_CONTEXT_LENGTHS = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384,
    'gpt-4-turbo-preview': 128000,
}

EMBEDDING_DIMENSIONS = {
    'thenlper/gte-base': 768,
    'thenlper/gte-large': 1024,
    'BAAI/bge-large-en': 1024,
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'gte-large-fine-tuned': 1024
}

SYS_PROMPT = "Answer the `query` using the `context` provided below. Be succinct. " \
"Feel free to ignore any contexts in the list that don't seem relevant to the givem query. " \
"If the question cannot be answered using the information provided, answer with 'I can't find any relevant context about {query}."
