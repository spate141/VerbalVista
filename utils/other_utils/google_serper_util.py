from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentType, load_tools, initialize_agent
from utils.rag_utils.summary_util import initialize_summarization_chain
from utils.data_parsing_utils.url_parser import parse_url


def google_serper_summarization(
        search_query: str = None, num_results: int = 3, temperature: float = 0.0,
        model_name: str = "gpt-3.5-turbo", chain_type: str = "map_reduce",
        max_tokens: int = 512, msg=None
):
    # Use the Google Serper API and fetch news results for given search_query
    search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1")
    result_dict = search.results(search_query)

    results = []
    if not result_dict['news']:
        results.append({"title": None, "link": None, "summary": None})
    else:
        # Load URL data from the top X news search results
        for i, item in zip(range(num_results), result_dict['news']):
            data = parse_url(item['link'], msg, return_data=True)

            # Initialize summarization chain
            summarization_chain = initialize_summarization_chain(
                model_name=model_name, temperature=temperature, max_tokens=max_tokens,
                chain_type=chain_type
            )
            summary = summarization_chain.run(data)
            results.append({"title": item['title'], "link": item['link'], "summary": summary})
    return results
