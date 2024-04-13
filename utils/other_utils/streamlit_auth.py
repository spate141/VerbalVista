import hmac
import streamlit as st


def check_password():
    """
    Returns `True` if the user had the correct password.
    """
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.markdown("<h2>🔐 Enter your password:</h2>", unsafe_allow_html=True)
    st.text_input(
        "Password", placeholder="Enter Password!", type="password", on_change=password_entered,
        key="password", label_visibility='collapsed'
    )
    if "password_correct" in st.session_state:
        st.error("❗️Password Incorrect ❗️")
    return False

