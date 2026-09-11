import os
from typing import TypedDict, Annotated
import operator
import psycopg
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (AnyMessage, SystemMessage, HumanMessage, AIMessage)
from langchain_groq import ChatGroq
from mcp_client import (tavily_mcp_search,
                        get_airports,get_airlines,
                        aviation_mcp_call,extract_destination,
                        forecast_mcp_search,weather_mcp_search)

llm = ChatGroq(model="openai/gpt-oss-20b")
DATABASE_URL=os.getenv("DATABASE_URL")

#state
class TravelState(TypedDict):    #state for all the agents using langgraph
    messages: Annotated[list[AnyMessage],operator.add] #messages key is used to store the conversation history between the user and the agents. It is a list of AnyMessage objects, which can be either SystemMessage, HumanMessage, or AIMessage. The operator.add annotation indicates that new messages will be appended to the existing list of messages.
    user_query: str              #user_query key is used to store the query that the user inputs in the chat interface. It is a string that represents the user's request or question related to travel, such as "Find me flights from New York to London" or "What are the best hotels in Paris?".
    flight_results: str          #flight_results key is used to store the results of the flight search. It is a string that contains information about available flights based on the user's query, such as airline names, departure and arrival airports, and flight status.
    hotel_results: str           #hotel_results key is used to store the results of the hotel search. It is a string that contains information about available hotels based on the user's query, such as hotel names, locations, ratings, and prices.
    itinerary: str               #itinerary key is used to store the generated travel itinerary. It is a string that outlines the recommended travel plan based on the user's query and the results from flight and hotel searches, including suggested flights, accommodations, and activities.
    llm_calls: int               #llm_calls key is used to store the number of times the LLM has been called during the execution of the graph. It is an integer that tracks the usage of the language model.
    weather_results:str


FLIGHT_AGENT_PROMPT="""you are a travel flight expert.
User Query:{query}
Airport Information:{airport_data}
Airline Information:{airline_data}

Generate:
1.Likely departure airport
2.Likely arrival airport
3.Airlines serving this route
4.Typical flight duration
5.Estimated airfare range
6.Peak season pricing warning
7.Booking advice

Return concise travel guidance."""


def flight_agent(state: TravelState):
    query = state["user_query"]
    try:
        airports=asyncio.run(
            aviation_mcp_call("list_airports")
        )
    
        airlines=asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        prompt=FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )


        response=llm.invoke([SystemMessage(content="you are an expert travel flight planner."),
                             HumanMessage(content=prompt)])

        flight_data= response.content

    except Exception as e:

        flight_data=f"Flight information unavailable:{str(e)}"


    return {
        "flight_results": flight_data,
        "messages":[AIMessage(content="Flight recommendation generated")],
        "llm_calls": state.get("llm_calls",0)+1
    }


def hotel_agent(state: TravelState):
    query = f"best hotels for {state['user_query']}"
    hotel_results = asyncio.run(
        tavily_mcp_search(query)
    )
    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="hotel info fetched")],
    }



def weather_agent(state:TravelState):
    city=extract_destination(state["user_query"])
    weather_data=asyncio.run(
        weather_mcp_search(city)
    )
    forecast_data=asyncio.run(
        forecast_mcp_search(city)
    )

    return{
        "weather_results": f"""current weather:{weather_data}
                               forecast:{forecast_data}""",
                               "messages":[AIMessage(content="weather information fetched"
            )
        ]           
    }



def itinerary_agent(state: TravelState):
    prompt = f"""Create a travel itinerary.
    User Query: {state['user_query']}
    Flight Results: {state['flight_results']}
    Hotel Results: {state['hotel_results']}
    Weather Information:{state['weather_results']}"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner"),
        HumanMessage(content=prompt),
    ])
    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def final_agent(state: TravelState):
    final_prompt = f"""Generate the final, polished travel response for the user.
    Flight: {state['flight_results']}
    Hotel: {state['hotel_results']}
    Weather: {state['weather_results']}
    Itinerary: {state['itinerary']}"""
    response = llm.invoke([HumanMessage(content=final_prompt)])
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }



graph = StateGraph(TravelState)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)


graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent",END)



def build_app():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # config = {"configurable": {"thread_id": "user_sougata"}}  
    import uuid  #everytime itll start afresh
    config={
        "configurable":{
            "thread_id": str(uuid.uuid4())
        }
    }


    user_input = input("enter travel request:")
    app = build_app()
    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    print("\nFINAL RESPONSE:\n")
    for msg in result["messages"]:
        print(msg.content)




