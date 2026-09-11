# AI Travel Booking System

A multi-agent AI travel planner built with **LangGraph**, **Groq**, and the **Model Context Protocol (MCP)**. Give it a single natural-language trip request and it coordinates a pipeline of specialized agents — flights, hotels, weather, itinerary, and a final summarizer — to produce a complete, ready-to-use travel plan.

## How it works

The system is modeled as a directed graph of agents, each responsible for one part of the trip:

```
START → Flight Agent → Hotel Agent → Weather Agent → Itinerary Agent → Final Agent → END
```

| Agent | Responsibility |
|---|---|
| **Flight Agent** | Queries live airport/airline data via the AviationStack MCP server and asks the LLM for likely routes, airlines, typical fares, and booking advice. |
| **Hotel Agent** | Uses the Tavily Search MCP server to pull current, real hotel recommendations and reviews for the destination. |
| **Weather Agent** | Calls a custom FastMCP weather server (backed by OpenWeather) for current conditions and a short-term forecast at the destination. |
| **Itinerary Agent** | Synthesizes flight, hotel, and weather data into a structured day-by-day plan. |
| **Final Agent** | Produces the polished, user-facing summary combining everything above. |

Each agent updates a shared `TravelState`, and the whole run is checkpointed to **PostgreSQL** via `langgraph.checkpoint.postgres`, so conversations can be resumed by `thread_id`.

## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — orchestrates the multi-agent pipeline as a stateful graph
- **[Groq](https://groq.com/)** (`openai/gpt-oss-20b`) — LLM powering each agent's reasoning
- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)** — connects agents to external tools:
  - [Tavily](https://tavily.com/) MCP server for web/hotel search
  - AviationStack MCP server for flight/airport/airline data
  - A custom FastMCP weather server wrapping the OpenWeather API
- **PostgreSQL** — conversation state persistence via `psycopg` + `PostgresSaver`
- **Streamlit** — interactive frontend with live per-agent status, a final plan view, and downloadable/auto-saved Markdown trip plans

## Project structure

```
.
├── main.py                          # LangGraph pipeline definition (agents, edges, graph build)
├── mcp_client.py                    # MCP server connections and tool wrappers
├── frontend.py                      # Streamlit UI
├── custom_weather_mcp_server.py     # Custom FastMCP weather server (OpenWeather)
├── testing_aviationstack_mcp_server.py  # Standalone tool-discovery script for AviationStack MCP
├── testing_weather_mcp_server.py    # Standalone tool-discovery script for the weather MCP server
└── .env                             # API keys and DB connection string (not committed)
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATION_STACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
DATABASE_URL=postgresql://user:password@host:port/dbname
```

### 3. Set up PostgreSQL

Ensure a reachable Postgres instance is available at `DATABASE_URL`. The checkpointer will automatically create the tables it needs (`checkpointer.setup()`) on first run.

### 4. Run

**Command line:**
```bash
python main.py
```

**Streamlit UI:**
```bash
streamlit run frontend.py
```

## Example

> "Plan a complete 7-day Japan trip including flights, hotels, and sightseeing under ₹2 lakhs"

The system will stream live status updates from each agent, then present a full itinerary with flight guidance, hotel suggestions, weather-aware planning, and a final consolidated trip summary you can download or find auto-saved under `travel_plans/`.

## Notes

- MCP tool responses are unwrapped and parsed before being passed between agents, so downstream agents work with clean data rather than raw MCP content blocks.
- Each conversation thread (`thread_id`) is checkpointed independently, allowing multiple users or sessions to run in parallel without state collisions.

## License

MIT
