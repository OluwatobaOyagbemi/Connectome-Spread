"""Entry point: landing page first, simulator one click away."""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Connectome Spread", page_icon="🧠", layout="wide",
                   initial_sidebar_state="auto")


def home():
    html = (Path(__file__).parent / "docs" / "index.html").read_text()
    # Inside Streamlit, the hero button goes to the simulator page, not the external URL.
    html = html.replace(
        'href="https://connectome-spread.streamlit.app"',
        'href="/simulator" onclick="window.top.location.href=\'/simulator\';return false;"')
    st.markdown("""
        <style>
          .block-container { padding: 0 !important; max-width: 100% !important; }
          header[data-testid="stHeader"] { background: transparent; }
          section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] { display: none; }
          iframe { border: 0; }
        </style>""", unsafe_allow_html=True)
    st.iframe(html, height=2450)


def simulator():
    import simulator as sim  # noqa: F401  (module runs the app on import)


pages = [
    st.Page(home, title="Home", url_path="home", default=True),
    st.Page(simulator, title="Simulator", url_path="simulator"),
]
st.navigation(pages, position="hidden").run()
