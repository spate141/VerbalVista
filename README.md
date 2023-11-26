<p align="center">
  <img align="center" src="https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png" width="40%" height="40%" />
</p>

![Static Badge](https://img.shields.io/badge/VerbalVista-1.3-blue)

## Streamlit Cloud:
- [VerbalVista](https://verbalvista.streamlit.app/)

## Set the keys:
- [Best Practices for API Key Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
- Edit the .env file at `PATH/TO/VerbalVista/.env` and set following keys
```dotenv
# Reddit
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=

# OpenAI
OPENAI_API_KEY=

# Serper
SERPER_API_KEY=
```

## Docker

### Build docker image:
```cmd
>> cd VerbalVista
>> docker build -t verbal_vista:1.3 .
```

### Start the docker image:
```cmd
>> docker run -p 8501:8501 verbal_vista:1.3
```

## Streamlit APP

### Start the program:
```cmd
>> streamlit run main.py
```

## Ray Server Management Script

This README provides guidance on using the `manage_server.sh` script for managing the Ray and application servers in your environment.

### Overview

The `manage_server.sh` script is a utility for easily starting and stopping the Ray server and a custom application server. It streamlines the process of server management through simple command-line commands.

### Prerequisites

- Ensure you have Ray installed in your environment.
- The script should be executable. You can set it as executable using `chmod +x manage_server.sh`.

### Usage

Run the script from the command line with the desired command. The script supports the following commands:

1. **start_ray**: Starts the Ray server.
```bash
bash manage_server.sh start_ray
```
2. **start_server [directory]**: Starts the application server with a specified index directory.
```bash
bash manage_server.sh start_server <index_directory>
```
3. **stop_server**: Stops the application server.
```bash
bash manage_server.sh stop_server
```
4. **stop_ray**: Stops the Ray server.
```bash
bash manage_server.sh stop_ray
```

### Ray server API endpoint example:
```cmd
curl --location 'http://127.0.0.1:8000/query' \
--header 'Content-Type: application/json' \
--data '{
    "query": "What is this document all about?",
    "llm": "gpt-3.5-turbo",
    "embedding_model": "text-embedding-ada-002",
    "temperature": 0.5,
    "max_semantic_retrieval_chunks": 5,
    "max_lexical_retrieval_chunks": 1
}
'
```
### Ray server sample output:
```cmd
{
    "query": "What is this document all about?",
    "answer": "The document is about Sqids, an open-source library that generates short, YouTube-looking IDs from numbers. These IDs can be customized and are collision-free. Sqids is mainly used for visual purposes, such as using IDs instead of numbers in web applications. It can be used for link shortening, event IDs, and generating one-time passwords.",
    "llm_model": "gpt-3.5-turbo",
    "embedding_model": "text-embedding-ada-002",
    "temperature": 0.5,
    "sources": [
        "sqids-org-.data.txt",
        "sqids-org-.data.txt",
        "sqids-org-.data.txt",
        "sqids-org-.data.txt"
    ],
    "completion_meta": {
        "completion_tokens": 69,
        "prompt_tokens": 873,
        "total_tokens": 942,
        "total_cost": {
            "completion": 0.00013800000000000002,
            "prompt": 0.0013095000000000001,
            "total": 0.0014475000000000002
        }
    }
}
```

## Available functions:
  - Media processing
  - Explore documents
  - Manage index
  - Q&A using LLM
  - Tell me about!
  - Stocks performance comparison
