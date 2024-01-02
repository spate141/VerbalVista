MAX_CONTEXT_LENGTHS = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384,
    'gpt-4-1106-preview': 128000
}

EMBEDDING_DIMENSIONS = {
    'thenlper/gte-base': 768,
    'thenlper/gte-large': 1024,
    'BAAI/bge-large-en': 1024,
    'text-embedding-ada-002': 1536,
    'gte-large-fine-tuned': 1024
}

SYS_PROMPT = "Answer the query using the context provided. Be succinct. " \
"Contexts are organized in a list of dictionaries [{'text': <context>}, {'text': <context>}, ...]. " \
"Feel free to ignore any contexts in the list that don't seem relevant to the query. " \
"If the question cannot be answered using the information provided, answer with 'I don't know'."
