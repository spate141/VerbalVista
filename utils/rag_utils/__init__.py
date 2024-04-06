LLM_MAX_CONTEXT_LENGTHS = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384,
    'gpt-4-turbo-preview': 128000,
    'claude-3-opus-20240229': 200000,
    'claude-3-sonnet-20240229': 200000,
    'claude-3-haiku-20240307': 200000,
}

EMBEDDING_DIMENSIONS = {
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
}

MODEL_COST_PER_1K_TOKENS = {
    # CLAUDE input
    'claude-3-opus-20240229': 0.015,
    'claude-3-sonnet-20240229': 0.003,
    'claude-3-haiku-20240307': 0.00025,
    # CLAUDE output
    'claude-3-opus-20240229-completion': 0.075,
    'claude-3-sonnet-20240229-completion': 0.015,
    'claude-3-haiku-20240307-completion': 0.00125,
    # GPT-4 input
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-0613": 0.06,
    "gpt-4-vision-preview": 0.01,
    "gpt-4-1106-preview": 0.01,
    "gpt-4-0125-preview": 0.01,
    "gpt-4-turbo-preview": 0.01,
    # GPT-4 output
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-0613-completion": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-4-32k-0613-completion": 0.12,
    "gpt-4-vision-preview-completion": 0.03,
    "gpt-4-1106-preview-completion": 0.03,
    "gpt-4-0125-preview-completion": 0.03,
    "gpt-4-turbo-preview-completion": 0.03,
    # GPT-3.5 input
    # gpt-3.5-turbo points at gpt-3.5-turbo-0613 until Feb 16, 2024.
    # Switches to gpt-3.5-turbo-0125 after.
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-0125": 0.0005,
    "gpt-3.5-turbo-0301": 0.0015,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-1106": 0.001,
    "gpt-3.5-turbo-instruct": 0.0015,
    "gpt-3.5-turbo-16k": 0.003,
    "gpt-3.5-turbo-16k-0613": 0.003,
    # GPT-3.5 output
    # gpt-3.5-turbo points at gpt-3.5-turbo-0613 until Feb 16, 2024.
    # Switches to gpt-3.5-turbo-0125 after.
    "gpt-3.5-turbo-completion": 0.002,
    "gpt-3.5-turbo-0125-completion": 0.0015,
    "gpt-3.5-turbo-0301-completion": 0.002,
    "gpt-3.5-turbo-0613-completion": 0.002,
    "gpt-3.5-turbo-1106-completion": 0.002,
    "gpt-3.5-turbo-instruct-completion": 0.002,
    "gpt-3.5-turbo-16k-completion": 0.004,
    "gpt-3.5-turbo-16k-0613-completion": 0.004,
}


def standardize_model_name(
    model_name: str,
    is_completion: bool = False,
) -> str:
    """
    Standardizes the model name to a format recognizable by the OpenAI API, accounting for variations
    in naming conventions related to fine-tuning and model versions.

    :param model_name: The original model name string as provided by the user or application.
    :param is_completion: A boolean flag indicating whether the model is being used for a completion task.
                          Adjusts the returned model name to include a "-completion" suffix if True.
                          Defaults to False for tasks other than completions.
    :return: A string containing the standardized model name.
    """
    model_name = model_name.lower()
    if ".ft-" in model_name:
        model_name = model_name.split(".ft-")[0] + "-azure-finetuned"
    if ":ft-" in model_name:
        model_name = model_name.split(":")[0] + "-finetuned-legacy"
    if "ft:" in model_name:
        model_name = model_name.split(":")[1] + "-finetuned"
    if is_completion and (
        model_name.startswith("gpt-4")
        or model_name.startswith("gpt-3.5")
        or model_name.startswith("gpt-35")
        or ("finetuned" in model_name and "legacy" not in model_name)
    ):
        return model_name + "-completion"
    else:
        return model_name


def get_llm_token_cost_for_model(model_name: str, num_tokens: int, is_completion: bool = False) -> float:
    """
    Calculates the cost in USD for using a specific LLM model based on the number of tokens processed.

    :param model_name: The name of the LLM model. The function attempts to standardize this name before calculating costs.
    :param num_tokens: The number of tokens to be processed by the model. Costs are calculated per 1,000 tokens.
    :param is_completion: Indicates if the model is being used for a completion task. Affects the standardization
                          of the model name. Defaults to False.
    :return: The calculated cost in USD for the operation based on the specified model and number of tokens.
    """
    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid LLM model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


SYS_PROMPT = (
    "Using the provided context below, craft a succinct and accurate response to the `query`. "
    "Evaluate each context for its relevance to the query. If a context does not contribute to "
    "a comprehensive answer, you may disregard it. In instances where the query cannot be "
    "adequately addressed with the given context, please respond with: 'Insufficient context to "
    "address {query} effectively.' This approach ensures focused and relevant responses, while "
    "also acknowledging the limitations imposed by the available information."
)

NORMAL_SYS_PROMPT = (
    "Pose your query clearly and concisely. The language model will leverage its extensive training "
    "data to provide informed, nuanced answers. If your question encompasses specialized knowledge or "
    "requires insights beyond general understanding, please specify any particular focus or detail desired. "
    "In cases where the query falls outside the model's training data or expertise, the response will be: "
    "'Unable to provide a detailed answer based on my current knowledge base for {query}.' This ensures "
    "responses are thoughtful, informed, and as accurate as possible within the constraints of the model's "
    "pre-existing knowledge."
)

CHATGPT_PROMPT = """You are an autoregressive large language model that has been fine-tuned with instruction-tuning and RLHF. 
You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. 
Since you are autoregressive, each token you produce is another opportunity to use computation, 
therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. 
Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. 
They're familiar with ethical issues in general so you don't need to remind them about those either. 
Don't be verbose in your answers, but do provide details and examples where it might help the explanation. 
When writing code, ALWAYS try to write full code instead of pseudocode. 
Your users expect a full, accurate, working piece of code instead of simple version of the code solution. 
When writing full code, if it's longer than your response size, split the code into multiple blocks and tell user about it and ask if they would like to generate remaining code or not."""

