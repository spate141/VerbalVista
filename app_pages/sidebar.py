from PIL import Image
import streamlit as st


def render_sidebar(app_name: str = None, app_version: str = None, app_pages: list = None):
    """
    Render app sidebar.
    :param app_name: Streamlit application name.
    :param app_version: Streamlit application version.
    :param app_pages: Streamlit application pages.
    """
    page_icon = Image.open('docs/logo-white.png')
    st.set_page_config(
        page_title=app_name,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/spate141/VerbalVista',
            'Report a bug': "https://github.com/spate141/VerbalVista",
            'About': "### Welcome to VerbalVista!\nBuilt by Snehal Patel."
        }
    )
    with st.sidebar:
        st.markdown(
            f"""
            <center>
            <a href="https://github.com/spate141/VerbalVista"><img src="https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png" width="50%" height="50%"></a>
            </br>
            </br>
            <h5 style="color: #233565">Version: {app_version}</h5>
            </center>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<center><h4><b>Select Page</b></h4></center>", unsafe_allow_html=True)
        selected_page = st.selectbox("Select function:", app_pages, label_visibility="collapsed")
        st.markdown("--------", unsafe_allow_html=True)
        # st.markdown("<h5><b>LLM API Key:</b></h5>", unsafe_allow_html=True)
        # llm_api_key = st.text_input(
        #     "LLM API Key:", key="chatbot_api_key", type="password", label_visibility="collapsed",
        #     placeholder="LLM API Key"
        # )
        return selected_page
