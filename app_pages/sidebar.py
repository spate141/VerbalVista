from PIL import Image
import streamlit as st
from typing import Optional, List


def render_sidebar(
    app_name: Optional[str] = None,
    app_version: Optional[str] = None,
    app_pages: Optional[List[str]] = None
) -> str:
    """
    Renders the sidebar for a Streamlit application, including the application's logo, name, version, and
    navigation for different pages within the app. It also provides links for help, bug reporting, and about
    information.

    :param app_name: Optional; The name of the Streamlit application.
    :param app_version: Optional; The version of the Streamlit application.
    :param app_pages: Optional; A list of pages (as strings) available for navigation in the app.
    :return: The name of the selected page from the sidebar.
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
    # SVG Editor: https://deeditor.com/
    # Original: https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png, #233565
    # Grey: https://i.ibb.co/GJmvgP7/download-1.png, #777777
    # Blue: https://i.ibb.co/t4sp8k3/download.png, #0099FF
    with st.sidebar:
        st.markdown(
            f"""
            <center>
            <a href="https://github.com/spate141/VerbalVista"><img src="https://i.ibb.co/6FQPs5C/verbal-vista-blue-transparent.png" width="60%" height="60%"></a>
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
        return selected_page
