<p align="center">
  <img align="center" src="https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png" width="40%" height="40%" />
</p>

![Static Badge](https://img.shields.io/badge/VerbalVista-1.1-blue)

### Streamlit Cloud:
- [VerbalVista](https://verbalvista.streamlit.app/)

### Set the keys:
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

### Build Docker Image [OPTIONAL]:
```cmd
>> cd VerbalVista
>> docker build -t verbal_vista:1.1 .
```

### Start the program:
```cmd
>> streamlit run main.py
```

_...or..._

```cmd
>> docker run -p 8501:8501 verbal_vista:1.1
```
  
### Available functions:
  - Media processing
  - Explore documents
  - Manage index
  - Q&A using LLM
  - Tell me about!
  - Stocks performance comparison

### Screenshots:

<details>
<summary>Media processing</summary>

![Screenshot 2023-07-16 at 4.31.08 PM.png](docs/Screenshot%202023-07-16%20at%204.31.08%20PM.png)

</details>

<details>
<summary>Explore documents</summary>

![Screenshot 2023-07-16 at 4.31.44 PM.png](docs/Screenshot%202023-07-16%20at%204.31.44%20PM.png)

</details>

<details>
<summary>Manage index</summary>

![Screenshot 2023-07-16 at 4.31.51 PM.png](docs/Screenshot%202023-07-16%20at%204.31.51%20PM.png)

</details>

<details>
<summary>Question/Answering</summary>

![Screenshot 2023-07-16 at 4.35.12 PM.png](docs/Screenshot%202023-07-16%20at%204.35.12%20PM.png)

</details>