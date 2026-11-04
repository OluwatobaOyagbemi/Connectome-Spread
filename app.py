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


pages = [
    st.Page(home, title="Home", default=True),   # default page is always served at "/"
    # A script path, not an import: Streamlit re-executes the file on every rerun.
    # (Importing it would run the module once per process and leave later
    # visitors with a blank page.)
    st.Page(str(Path(__file__).parent / "simulator.py"), title="Simulator",
            url_path="simulator"),
]
st.navigation(pages, position="hidden").run()
