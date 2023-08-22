from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentType, load_tools, initialize_agent
from utils.document_parser import parse_url


class GoogleSerperUtil:

    def __init__(self):
        pass

    @staticmethod
    def load_google_serper_agent(search_query: str = None, temperature: float = 0.0, verbose: bool = True):
        """

        """
        llm = OpenAI(temperature=temperature)
        tools = load_tools(["google-serper"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)
        answer = agent.run(search_query)

        # message_placeholder = st.empty()
        # # Simulate stream of response with milliseconds delay
        # full_response = ""
        # for chunk in answer.split():
        #     full_response += chunk + " "
        #     time.sleep(0.03)
        #     # Add a blinking cursor to simulate typing
        #     message_placeholder.markdown(full_response + "▌")
        # # Display full message at the end with other stuff you want to show like `response_meta`.
        # message_placeholder.markdown(full_response)

        return answer

    @staticmethod
    def google_serper_summarization(
            search_query: str = None, num_results: int = 3, temperature: float = 0.0,
            model_name: str = "gpt-3.5-turbo", chain_type: str = "map_reduce",
            max_tokens: int = 512, summary_util=None
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
                data = parse_url(item['link'], return_data=True)

                # Initialize summarization chain
                summarization_chain = summary_util.initialize_summarization_chain(
                    model_name=model_name, temperature=temperature, max_tokens=max_tokens,
                    chain_type=chain_type
                )
                summary = summarization_chain.run(data)
                results.append({"title": item['title'], "link": item['link'], "summary": summary})
        return results
