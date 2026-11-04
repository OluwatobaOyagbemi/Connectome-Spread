"""Entry point: landing page first, simulator one click away."""
import base64
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Connectome Spread", page_icon="🧠", layout="wide",
                   initial_sidebar_state="auto")


def home():
    html = (Path(__file__).parent / "docs" / "index.html").read_text()
    # Rendered directly in the Streamlit document, not in an iframe: Streamlit's
    # iframe sandbox blocks top-level navigation, so links to GitHub showed
    # "refused to connect" and the hero button could not reach /simulator.
    # st.html strips <script>, so start the arrival animation via CSS instead
    # and hide the Replay button.
    body = html[html.index("<body>") + 6: html.index("</body>")]
    style = html[html.index("<style>"): html.index("</style>") + 8]
    body = body.replace('href="https://connectome-spread.streamlit.app"', 'href="/simulator"')
    body = body.replace('<script>', '<!--').replace('</script>', '-->')
    # The sanitiser also strips inline SVG children, so ship the brain graph as
    # an <img> whose SVG carries its own styles and starts animating on load.
    i, j = body.index("<svg"), body.index("</svg>") + 6
    svg = body[i:j].replace(
        ">", ">" + "<style>"
        ".edges line{stroke:#13213a}"
        ".n{fill:#cfe0f7;stroke:#fff;stroke-width:1.5}.n.dim{fill:#e3e8ee}"
        ".n.lit{animation:arrive .9s ease-out forwards}"
        "@keyframes arrive{0%{fill:#cfe0f7}60%{fill:#e34948}100%{fill:#2a78d6}}"
        ".hemi{font:12px system-ui,sans-serif;fill:#7b8794}"
        "</style>", 1)
    data = base64.b64encode(svg.encode()).decode()
    body = body[:i] + (f'<img src="data:image/svg+xml;base64,{data}" alt="68 brain regions '
                       'lighting up in simulated arrival order" style="width:100%;height:auto;'
                       'display:block">') + body[j:]
    st.html("""
        <style>
          .block-container { padding: 0 !important; max-width: 100% !important; }
          header[data-testid="stHeader"] { background: transparent; }
          section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] { display: none; }
          .stApp { background: #eef2f6; }
          #replay { display: none; }
        </style>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    """ + style + body)


pages = [
    st.Page(home, title="Home", default=True),   # default page is always served at "/"
    # A script path, not an import: Streamlit re-executes the file on every rerun.
    # (Importing it would run the module once per process and leave later
    # visitors with a blank page.)
    st.Page(str(Path(__file__).parent / "simulator.py"), title="Simulator",
            url_path="simulator"),
]
st.navigation(pages, position="hidden").run()
