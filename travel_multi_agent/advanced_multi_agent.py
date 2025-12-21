#!/usr/bin/env -S uv run --with-requirements requirements.txt
"""
Advanced Multi-Agent Travel Planning System

This script demonstrates a production-ready multi-agent system using LangGraph with:
- Multiple specialized agents (Itinerary, Flight, Hotel)
- Tool integration (Tavily search, SERP API for flights/hotels)
- Smart routing between agents
- Checkpoint memory for multi-turn conversations

Prerequisites:
    1. Install dependencies: pip install -r requirements.txt
    2. Create a .env file in the project root with:
       OPENAI_API_KEY=your_key_here
       TAVILY_API_KEY=your_key_here
       SERPAPI_API_KEY=your_key_here
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional
import operator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
import serpapi


# ============================================================================
# State Schema
# ============================================================================

class TravelPlannerState(TypedDict):
    """Simple state schema for travel multiagent system"""
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: Optional[str]
    user_query: Optional[str]


# ============================================================================
# Tool Functions
# ============================================================================

def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: str = None,
    adults: int = 1,
    children: int = 0
) -> str:
    """
    Search for flights using Google Flights engine.

    Args:
        departure_airport: Departure airport code (e.g., 'NYC', 'LAX')
        arrival_airport: Arrival airport code (e.g., 'LON', 'NRT')
        outbound_date: Departure date (YYYY-MM-DD format)
        return_date: Return date (YYYY-MM-DD format, optional for one-way)
        adults: Number of adult passengers (default: 1)
        children: Number of child passengers (default: 0)
    """
    adults = int(float(adults)) if adults else 1
    children = int(float(children)) if children else 0

    params = {
        'api_key': os.getenv('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'hl': 'en',
        'gl': 'us',
        'departure_id': departure_airport,
        'arrival_id': arrival_airport,
        'outbound_date': outbound_date,
        'currency': 'USD',
        'adults': adults,
        'children': children,
        'type': '2' if not return_date else '1'
    }

    if return_date:
        params['return_date'] = return_date

    try:
        search = serpapi.search(params)
        results = search.data.get('best_flights', [])
        if not results:
            results = search.data.get('other_flights', [])
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Flight search failed: {str(e)}"


def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
    hotel_class: str = None,
    sort_by: int = 8
) -> str:
    """
    Search for hotels using Google Hotels engine.

    Args:
        location: Location to search for hotels (e.g., 'New York', 'Paris', 'Tokyo')
        check_in_date: Check-in date (YYYY-MM-DD format)
        check_out_date: Check-out date (YYYY-MM-DD format)
        adults: Number of adults (default: 1)
        children: Number of children (default: 0)
        rooms: Number of rooms (default: 1)
        hotel_class: Hotel class filter (e.g., '2,3,4' for 2-4 star hotels)
        sort_by: Sort parameter (default: 8 for highest rating)
    """
    adults = int(float(adults)) if adults else 1
    children = int(float(children)) if children else 0
    rooms = int(float(rooms)) if rooms else 1
    sort_by = int(float(sort_by)) if sort_by else 8

    params = {
        'api_key': os.getenv('SERPAPI_API_KEY'),
        'engine': 'google_hotels',
        'hl': 'en',
        'gl': 'us',
        'q': location,
        'check_in_date': check_in_date,
        'check_out_date': check_out_date,
        'currency': 'USD',
        'adults': adults,
        'children': children,
        'rooms': rooms,
        'sort_by': sort_by
    }

    if hotel_class:
        params['hotel_class'] = hotel_class

    try:
        search = serpapi.search(params)
        properties = search.data.get('properties', [])
        if not properties:
            return f"No hotels found. Available data keys: {list(search.data.keys())}"
        return json.dumps(properties[:5], indent=2)
    except Exception as e:
        return f"Hotel search failed: {str(e)}"


# ============================================================================
# Prompt Templates
# ============================================================================

def get_itinerary_prompt():
    """Get itinerary agent prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert travel itinerary planner. ONLY respond to travel planning and itinerary-related questions.

IMPORTANT RULES:
- If asked about non-travel topics (weather, math, general questions), politely decline and redirect to travel planning
- Always provide complete, well-formatted itineraries with specific details
- Include timing, locations, transportation, and practical tips

Use the ReAct approach:
1. THOUGHT: Analyze what travel information is needed
2. ACTION: Search for current information about destinations, attractions, prices, hours
3. OBSERVATION: Process the search results
4. Provide a comprehensive, formatted response

Available tools:
- TavilySearch: Search for current travel information

Format your itineraries with:
- Clear day-by-day breakdown
- Specific times and locations
- Transportation between locations
- Estimated costs when possible
- Practical tips and recommendations"""),
        MessagesPlaceholder(variable_name="messages"),
    ])


def get_flight_prompt():
    """Get flight agent prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a flight booking expert. ONLY respond to flight-related queries.

IMPORTANT RULES:
- If asked about non-flight topics, politely decline and redirect to flight booking
- Always use the search_flights tool to find current flight information
- You CAN search for flights and analyze the results for:
  * Direct flights vs connecting flights
  * Different airlines and flight classes
  * Various price ranges and timing options
  * Flight duration and layover information
- When users ask for specific preferences (direct flights, specific class, etc.), search first then filter/analyze the results
- Present results clearly organized by outbound and return flights

Available tools:
- search_flights: Search for comprehensive flight data that includes all airlines, classes, and connection types

Process:
1. ALWAYS search for flights first using the tool
2. Analyze the results to find flights matching user preferences
3. Present organized results with clear recommendations

Airport code mapping:
- Delhi: DEL
- London Heathrow: LHR
- New York: JFK/LGA/EWR
- etc."""),
        MessagesPlaceholder(variable_name="messages"),
    ])


def get_hotel_prompt():
    """Get hotel agent prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a hotel booking expert. ONLY respond to hotel and accommodation-related queries.

IMPORTANT RULES:
- If asked about non-hotel topics, politely decline and redirect to hotel booking
- Always use the search_hotels tool to find current hotel information
- Provide detailed hotel options with prices, ratings, amenities, and location details
- Include practical booking advice and tips
- You CAN search and analyze results for different criteria like star ratings, price ranges, amenities

Available tools:
- search_hotels: Search for hotels using Google Hotels engine

When searching hotels, extract or ask for:
- Location/destination
- Check-in and check-out dates (YYYY-MM-DD format)
- Number of guests (adults, children)
- Number of rooms
- Hotel preferences (star rating, amenities, etc.)

Present results with:
- Hotel name and star rating
- Price per night and total cost
- Key amenities and features
- Location and nearby attractions
- Booking recommendations"""),
        MessagesPlaceholder(variable_name="messages"),
    ])


# ============================================================================
# Agent Node Builder Functions
# ============================================================================

def create_itinerary_agent_node(itinerary_agent, tavily_tool):
    """Create itinerary agent node function"""
    def itinerary_agent_node(state: TravelPlannerState):
        messages = state["messages"]
        response = itinerary_agent.invoke({"messages": messages})

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'tavily_search_results_json':
                    try:
                        tool_result = tavily_tool.search(query=tool_call['args']['query'], max_results=2)
                        tool_result = json.dumps(tool_result, indent=2)
                    except Exception as e:
                        tool_result = f"Search failed: {str(e)}"

                    tool_messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    ))

            if tool_messages:
                all_messages = messages + [response] + tool_messages
                final_response = itinerary_agent.invoke({"messages": all_messages})
                return {"messages": [response] + tool_messages + [final_response]}

        return {"messages": [response]}

    return itinerary_agent_node


def create_flight_agent_node(flight_agent):
    """Create flight agent node function"""
    def flight_agent_node(state: TravelPlannerState):
        messages = state["messages"]
        response = flight_agent.invoke({"messages": messages})

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_flights':
                    try:
                        tool_result = search_flights(**tool_call['args'])
                    except Exception as e:
                        tool_result = f"Flight search failed: {str(e)}"

                    tool_messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    ))

            if tool_messages:
                all_messages = messages + [response] + tool_messages
                final_response = flight_agent.invoke({"messages": all_messages})
                return {"messages": [response] + tool_messages + [final_response]}

        return {"messages": [response]}

    return flight_agent_node


def create_hotel_agent_node(hotel_agent):
    """Create hotel agent node function"""
    def hotel_agent_node(state: TravelPlannerState):
        messages = state["messages"]
        response = hotel_agent.invoke({"messages": messages})

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_hotels':
                    try:
                        tool_result = search_hotels(**tool_call['args'])
                    except Exception as e:
                        tool_result = f"Hotel search failed: {str(e)}"

                    tool_messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    ))

            if tool_messages:
                all_messages = messages + [response] + tool_messages
                final_response = hotel_agent.invoke({"messages": all_messages})
                return {"messages": [response] + tool_messages + [final_response]}

        return {"messages": [response]}

    return hotel_agent_node


def create_router(llm):
    """Creates a router for the three travel agents"""
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing expert for a travel planning system.

        Analyze the user's query and decide which specialist agent should handle it:

        - FLIGHT: Flight bookings, airlines, air travel, flight search, tickets, airports, departures, arrivals, airline prices
        - HOTEL: Hotels, accommodations, stays, rooms, hotel bookings, lodging, resorts, hotel search, hotel prices
        - ITINERARY: Travel itineraries, trip planning, destinations, activities, attractions, sightseeing, travel advice, weather, culture, food, general travel questions

        Respond with ONLY one word: FLIGHT, HOTEL, or ITINERARY

        Examples:
        "Book me a flight to Paris" → FLIGHT
        "Find hotels in Tokyo" → HOTEL
        "Plan my 5-day trip to Italy" → ITINERARY
        "Search flights from NYC to London" → FLIGHT
        "Where should I stay in Bali?" → HOTEL
        "What are the best attractions in Rome?" → ITINERARY
        "I need airline tickets" → FLIGHT
        "Show me hotel options" → HOTEL
        "Create an itinerary for Japan" → ITINERARY"""),
        ("user", "Query: {query}")
    ])

    router_chain = router_prompt | llm | StrOutputParser()

    def route_query(state):
        """Router function for LangGraph - decides which agent to call next"""
        user_message = state["messages"][-1].content
        print(f"🧭 Router analyzing: '{user_message[:50]}...'")

        try:
            decision = router_chain.invoke({"query": user_message}).strip().upper()
            if decision not in ["FLIGHT", "HOTEL", "ITINERARY"]:
                decision = "ITINERARY"

            agent_mapping = {
                "FLIGHT": "flight_agent",
                "HOTEL": "hotel_agent",
                "ITINERARY": "itinerary_agent"
            }

            next_agent = agent_mapping.get(decision, "itinerary_agent")
            print(f"🎯 Router decision: {decision} → {next_agent}")
            return next_agent

        except Exception as e:
            print(f"⚠️ Router error, defaulting to itinerary_agent: {e}")
            return "itinerary_agent"

    return route_query


def create_router_node(router):
    """Create router node function"""
    def router_node(state: TravelPlannerState):
        user_message = state["messages"][-1].content
        next_agent = router(state)
        return {
            "next_agent": next_agent,
            "user_query": user_message
        }
    return router_node


def route_to_agent(state: TravelPlannerState):
    """Conditional edge function - routes to appropriate agent"""
    next_agent = state.get("next_agent")

    if next_agent == "flight_agent":
        return "flight_agent"
    elif next_agent == "hotel_agent":
        return "hotel_agent"
    elif next_agent == "itinerary_agent":
        return "itinerary_agent"
    else:
        return "itinerary_agent"


# ============================================================================
# System Initialization
# ============================================================================

def initialize_system():
    """Initialize all components of the travel planning system"""

    # Load environment variables
    load_dotenv()

    # Verify API keys
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}\n"
            f"Please add them to your .env file in the project root."
        )

    print("✅ API keys loaded successfully")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    print("✅ LLM initialized")

    # Initialize tools
    tavily_tool = TavilySearch(max_results=2)
    print("✅ Tavily search tool initialized")
    print("✅ Flight and hotel search tools initialized")

    # Create prompts
    itinerary_prompt = get_itinerary_prompt()
    flight_prompt = get_flight_prompt()
    hotel_prompt = get_hotel_prompt()

    # Bind tools to agents
    llm_with_tavily_tools = llm.bind_tools([tavily_tool])
    llm_with_flight_tools = llm.bind_tools([search_flights])
    llm_with_hotel_tools = llm.bind_tools([search_hotels])

    # Create agent chains
    itinerary_agent = itinerary_prompt | llm_with_tavily_tools
    flight_agent = flight_prompt | llm_with_flight_tools
    hotel_agent = hotel_prompt | llm_with_hotel_tools
    print("✅ All agents created")

    # Create router
    router = create_router(llm)
    print("✅ Travel Router created")

    # Create node functions
    itinerary_agent_node = create_itinerary_agent_node(itinerary_agent, tavily_tool)
    flight_agent_node = create_flight_agent_node(flight_agent)
    hotel_agent_node = create_hotel_agent_node(hotel_agent)
    router_node = create_router_node(router)

    # Build graph
    workflow = StateGraph(TravelPlannerState)
    workflow.add_node("router", router_node)
    workflow.add_node("flight_agent", flight_agent_node)
    workflow.add_node("hotel_agent", hotel_agent_node)
    workflow.add_node("itinerary_agent", itinerary_agent_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
            "itinerary_agent": "itinerary_agent"
        }
    )

    workflow.add_edge("flight_agent", END)
    workflow.add_edge("hotel_agent", END)
    workflow.add_edge("itinerary_agent", END)

    checkpointer = InMemorySaver()
    travel_planner = workflow.compile(checkpointer=checkpointer)

    print("✅ Travel Planning Graph built successfully!")

    return travel_planner


# ============================================================================
# Test Functions
# ============================================================================

def test_system(travel_planner, query, thread_id="test_thread"):
    """Test the multi-agent system with a single query"""
    print(f"\n{'='*60}")
    print(f"🧑 User: {query}")
    print(f"{'='*60}\n")

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": ""
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_planner.invoke(initial_state, config)

    response = result["messages"][-1].content
    print(f"\n🤖 Assistant: {response}")
    print("-" * 60)


def multi_turn_chat(travel_planner):
    """Multi-turn conversation with checkpoint memory"""
    print("\n" + "="*60)
    print("💬 Multi-Agent Travel Assistant (Multi-turn Mode)")
    print("="*60)
    print("Type 'quit' to exit\n")

    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("🧑 You: ")

        if user_input.lower() == 'quit':
            print("\n👋 Goodbye! Safe travels!")
            break

        print(f"\n📊 Processing query...\n")

        result = travel_planner.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )

        response = result["messages"][-1].content
        print(f"\n🤖 Assistant: {response}")
        print("-" * 60 + "\n")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function"""
    print("\n" + "="*60)
    print("ADVANCED MULTI-AGENT TRAVEL PLANNING SYSTEM")
    print("="*60 + "\n")

    # Initialize the system
    travel_planner = initialize_system()

    # Run test queries
    print("\n### Running Test Queries ###\n")
    test_system(travel_planner, "I need to book a flight to Dubai on 30 Nov 2025 from New York for 1 person")
    test_system(travel_planner, "Find me a good hotel in New Delhi on 27 Nov 2025 for 1 night for 1 adult")
    test_system(travel_planner, "What's the weather like in Paris?")

    # Start interactive chat
    print("\n### Starting Interactive Mode ###")
    multi_turn_chat(travel_planner)


if __name__ == "__main__":
    main()
