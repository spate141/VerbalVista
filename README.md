<p align="center">
  <img align="center" src="https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png" width="40%" height="40%" />
</p>

![Static Badge](https://img.shields.io/badge/VerbalVista-1.1-blue)

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
>> docker build -t verbal_vista:1.1 .
```

### Start the docker image:
```cmd
>> docker run -p 8501:8501 verbal_vista:1.1
```

## Streamlit APP

### Start the program:
```cmd
>> streamlit run main.py
```

## Ray server

### Start the ray server
```cmd
>> cd serving
>> ray start --head
>> python serve.py
```

### Stop the ray server
```cmd
>> ray stop
```

### Ray server API endpoint example:
```cmd
curl --location 'http://127.0.0.1:8000/query' \
--header 'Content-Type: application/json' \
--data '{
    "query": "What is going on?"
}'
```

## Available functions:
  - Media processing
  - Explore documents
  - Manage index
  - Q&A using LLM
  - Tell me about!
  - Stocks performance comparison
