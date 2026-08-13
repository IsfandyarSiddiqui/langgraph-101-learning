import sqlite3
from datetime import datetime
from pathlib import Path

import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


MARKDOWN_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "markdown"


def save_response_to_markdown(response, filename: str | None = None) -> Path:
    """Save an LLM or graph response as a Markdown file.

    Args:
        response: A string, a LangChain message with a ``content`` attribute,
            or a graph result containing a ``final_report`` value.
        filename: Output filename. A timestamped name is used when omitted.

    Returns:
        The absolute path of the saved Markdown file.
    """
    if isinstance(response, str):
        content = response
    elif isinstance(response, dict) and "final_report" in response:
        content = response["final_report"]
    elif hasattr(response, "content"):
        content = response.content
    else:
        raise TypeError(
            "response must be a string, a message with content, or a graph "
            "result containing final_report"
        )

    if not isinstance(content, str):
        raise TypeError("response content must be a string")

    if filename is None:
        filename = f"response-{datetime.now():%Y%m%d-%H%M%S}.md"

    output_name = Path(filename).name
    if not output_name.lower().endswith(".md"):
        output_name += ".md"

    MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MARKDOWN_OUTPUT_DIR / output_name
    output_path.write_text(content, encoding="utf-8")
    return output_path

def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with ASCII fallback.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
        xray: Whether to show the internal structure of the graph
    """
    from IPython.display import Image
    try:
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        print(f"⚠️  Image rendering failed: {e}")
        print("\n📊 Showing ASCII diagram instead:\n")
        ascii_diagram = graph.get_graph(xray=xray).draw_ascii()
        print(ascii_diagram)
        return None

def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
