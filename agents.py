from langgraph.prebuilt import create_react_agent,InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool,InjectedToolCallId
import sqlite3
from langgraph_supervisor import create_supervisor
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Tuple, Dict, Annotated,Any

load_dotenv()
checkpointer = InMemorySaver()
DB_PATH = "./ChinHook_database.db"

@tool
def catalog_search(genre: str) -> List[Tuple[int, str]]:
    """Search for up to 5 tracks matching a given genre in the Chinook database.

    Args:
        genre (str): The name of the genre to filter tracks by (case‐sensitive).

    Returns:
        List[Tuple[int, str]]: A list of (TrackId, Name) tuples for the first 5 matching tracks.

    Raises:
        ValueError: If `genre` is an empty string.
        sqlite3.Error: If there is any issue connecting to or querying the database.
    """
    if not genre:
        raise ValueError("`genre` must be a non-empty string.")

    query = "SELECT t.TrackId, t.Name FROM Track AS t JOIN Genre AS g ON t.GenreId = g.GenreId WHERE g.Name = ? LIMIT 5;"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(query, (genre,))
            rows = cursor.fetchall()
    except sqlite3.Error as e:
        # You might log this exception in a real system
        raise

    return rows

@tool
def database_search(query: str) -> List[Tuple[Any, ...]]:
    """
    Execute a user-provided SQL query against the Chinook database and return up to 100 results.

    This tool grants full, read-only visibility into all 11 tables in the Chinook DB
    (Artist, Album, Track, Genre, Customer, Invoice, InvoiceLine, MediaType, Playlist, Employee, PlaylistTrack).

    Use it for SELECT queries only. Long-running or destructive SQL commands will be rejected.

    Args:
        query (str): A valid SQL SELECT statement targeting one or more Chinook tables.

    Returns:
        List[Tuple[Any, ...]]: Up to the first 100 rows of results matching the query.

    Raises:
        ValueError: If the query is empty, non-SELECT, or appears destructive.
        sqlite3.Error: On SQL syntax or execution errors.
    """
    if not query or not query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(query + " LIMIT 100;")
            return cursor.fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {e}")


@tool
def create_order(customer_id: int, cart: List[Dict[str, float]]) -> int:
    """Create a new invoice and associated line items for a customer's cart.

    Args:
        customer_id (int): The ID of the customer placing the order.
        cart (List[Dict[str, float]]): A list of items, each a dict with keys:
            - "TrackId" (int): the track identifier
            - "UnitPrice" (float): price per unit
            - "Quantity" (int): number of units

    Returns:
        int: The newly created InvoiceId.

    Raises:
        ValueError: If `cart` is empty or contains invalid item entries.
        sqlite3.Error: If there is any issue connecting to or modifying the database.
    """
    if not cart:
        raise ValueError("`cart` must contain at least one item.")

    # Validate cart entries
    total = 0.0
    for item in cart:
        try:
            tid = int(item["TrackId"])
            qty = int(item["Quantity"])
            price = float(item["UnitPrice"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid cart item format: {item!r}")
        total += price * qty

    insert_invoice_sql = "INSERT INTO Invoice (CustomerId, Total) VALUES (?, ?);"
    insert_line_sql = (
        "INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity) "
        "VALUES (?, ?, ?, ?);"
    )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            # Create the invoice
            cur.execute(insert_invoice_sql, (customer_id, total))
            invoice_id = cur.lastrowid

            # Create each invoice line
            for item in cart:
                cur.execute(
                    insert_line_sql,
                    (
                        invoice_id,
                        item["TrackId"],
                        item["UnitPrice"],
                        item["Quantity"],
                    ),
                )
            conn.commit()
    except sqlite3.Error as e:
        # Optionally log e
        raise

    return invoice_id

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",max_retries=4,rate_limiter=InMemoryRateLimiter(requests_per_second=5.0) ,temperature=0)

system_catalog = (
    "You are **CatalogAgent**. "
    "When given a music genre, find up to 5 matching tracks from Chinook DB. "
    "Return JSON: [{TrackId: int, Name: str}, ...]."
)
system_order = (
    "You are **OrderAgent**. "
    "Given a customer_id and cart items list, "
    "create an invoice via the create_order tool and return JSON: {invoice_id: int}."
)

catalog_agent = create_react_agent(
    model=llm,
    name = "CatalogAgent",
    tools=[catalog_search],
    prompt=system_catalog
)

order_agent = create_react_agent(
    model=llm,
    name = "OrderAgent",
    tools=[create_order],
    prompt=system_order
)

PROMPT = """
You are **DBAgent for Chinook**—a virtual assistant with SQL access to the Chinook sample database.

This DB contains:
- 11 tables (Artist, Album, Track, Genre, Customer, Invoice, InvoiceLine, MediaType, Playlist, Employee, PlaylistTrack),
- Contains media store data (artists, tracks, customer invoices, etc.) :contentReference[oaicite:6]{index=6}.

Your task:
- Interpret the user's natural language request,
- Generate a valid SQL SELECT statement,
- Execute it via `database_search` tool,
- Return results as JSON or formatted lists.

*Important*: Only use the database_search tool—do not write SQL yourself in the agent.
"""

db_agent = create_react_agent(
    model=llm,
    tools=[database_search],
    name="ChinookDBAgent",
    prompt=PROMPT,
)


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool

handoff_to_catalog = create_handoff_tool(
    agent_name="CatalogAgent",
    description="Send a search-request to CatalogAgent."
)
handoff_to_order = create_handoff_tool(
    agent_name="OrderAgent",
    description="Send an order creation request to OrderAgent."
)
handoff_to_db = create_handoff_tool(
    agent_name="ChinookDBAgent",
    description="Send a database query request to ChinookDBAgent."
)

SUPERVISOR_PROMPT = """
You are the MusicStore Supervisor coordinating three agents:

1. CatalogAgent: Browses tracks by genre in the Chinook database.
2. OrderAgent: Creates orders (invoices) in the Chinook database.
3. ChinookDBAgent: Executes arbitrary SQL SELECT queries across all Chinook tables (Artist, Album, Track, Genre, Customer, Invoice, InvoiceLine, etc.).

When a user asks:
- To browse music, use the CatalogAgent.
- To submit an order, use the OrderAgent.
- To run a custom database query (SELECT only), use the ChinookDBAgent.

Always hand off to exactly one agent at a time using the appropriate handoff tool.
Once an agent returns its result, **you** must:
1. Summarize **all agents’ responses so far**, 
2. Compose a single, coherent final answer to the user that directly addresses their query, 
3. Do **not** perform any additional computation yourself.
"""

supervisor = create_supervisor(
    model=llm,
    agents=[catalog_agent,order_agent,db_agent],
    tools=[handoff_to_catalog,handoff_to_order,handoff_to_db],
    prompt=SUPERVISOR_PROMPT,
    supervisor_name ="MusicStoreSupervisor",
    add_handoff_back_messages=True,
).compile(checkpointer=checkpointer)


if __name__=="__main__":
    config = {"configurable": {"thread_id": "1"}}
    message= HumanMessage(content="How many invoices does customer 12 have?")
    for chunk in supervisor.stream({"messages":message},config=config,stream_mode="values"):
        for m in chunk["messages"]:
            m.pretty_print()