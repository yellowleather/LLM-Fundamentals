#!/usr/bin/env -S uv run --with-requirements requirements.txt
"""
Introduction to LangGraph - Simple Travel Agent Example

This script demonstrates the key concepts of LangGraph:
- StateGraph components
- Multiple connected agents
- State transitions

Prerequisites:
    pip install -r requirements.txt
"""

from typing import TypedDict
from langgraph.graph import StateGraph


# ============================================================================
# Step 1: Define the Shared State
# ============================================================================

class TravelState(TypedDict, total=False):
    """Shared state that every agent can read and update"""
    user_input: str          # raw question
    destination: str         # "Bali", "Manali", …
    itinerary: str           # multi-day plan
    activities: str          # granular activities


# ============================================================================
# Step 2: Create Three Simple Agents
# ============================================================================

def destination_agent(state: TravelState) -> TravelState:
    """LangGraph node for destination agent"""
    print("🔧 destination_agent running…")
    q = state.get("user_input", "").lower()

    if "beach" in q:
        dest = "Bali"
    elif "snow" in q or "mountain" in q:
        dest = "Manali"
    else:
        dest = "Kyoto"  # sensible default

    print(f"🌍 Suggested destination: {dest}")
    return {**state, "destination": dest}


def itinerary_agent(state: TravelState) -> TravelState:
    """LangGraph node for itinerary agent"""
    print("🔧 itinerary_agent running…")
    dest = state["destination"]
    plan = (
        f"Day 1: Arrive in {dest}\n"
        f"Day 2: Explore iconic spots in {dest}\n"
        f"Day 3: Relax + sample local cuisine"
    )
    print(f"🧳 Draft itinerary:\n{plan}")
    return {**state, "itinerary": plan}


def activity_agent(state: TravelState) -> TravelState:
    """LangGraph node for activity agent"""
    print("🔧 activity_agent running…")
    dest = state["destination"]
    act = (
        "Snorkelling • Beach yoga"
        if dest == "Bali"
        else "Skiing • Mountain trek"
    )
    print(f"🎯 Suggested activities: {act}")
    return {**state, "activities": act}


# ============================================================================
# Step 3: Build the LangGraph Workflow
# ============================================================================

def build_simple_graph():
    """Build a simple sequential travel planning graph"""
    builder = StateGraph(TravelState)

    # Add nodes
    builder.add_node("destination_agent", destination_agent)
    builder.add_node("itinerary_agent", itinerary_agent)
    builder.add_node("activity_agent", activity_agent)

    # Define flow: destination → itinerary → activity
    builder.set_entry_point("destination_agent")
    builder.add_edge("destination_agent", "itinerary_agent")
    builder.add_edge("itinerary_agent", "activity_agent")
    builder.set_finish_point("activity_agent")

    # Compile
    travel_graph = builder.compile()
    print("✅ Graph compiled.\n")

    return travel_graph


# ============================================================================
# Step 4: Run Examples
# ============================================================================

def run_example(graph, user_input: str):
    """Run the graph with a user input"""
    print(f"\n{'='*60}")
    print(f"User Query: {user_input}")
    print(f"{'='*60}\n")

    initial_state: TravelState = {"user_input": user_input}
    final_state = graph.invoke(initial_state)

    print("\n🏁 Final state:")
    for k, v in final_state.items():
        print(f"  {k}: {v}")
    print()


# ============================================================================
# Extended Example: Enhanced Agents with More Destinations
# ============================================================================

def new_destination_agent(state: TravelState) -> TravelState:
    """Enhanced destination agent with city break support"""
    print("🔧 new_destination_agent running…")
    q = state.get("user_input", "").lower()

    if "beach" in q:
        dest = "Bali"
    elif "snow" in q or "mountain" in q:
        dest = "Manali"
    elif "city break" in q or "city" in q:
        dest = "Paris"
    else:
        dest = "Kyoto"  # sensible default

    print(f"🌍 Suggested destination: {dest}")
    return {**state, "destination": dest}


def new_itinerary_agent(state: TravelState) -> TravelState:
    """Enhanced itinerary agent with Paris-specific plan"""
    print("🔧 new_itinerary_agent running…")
    dest = state["destination"]

    if dest == "Paris":
        plan = (
            f"Day 1: Arrive in Paris, visit Eiffel Tower\n"
            f"Day 2: Explore Louvre Museum and Notre Dame\n"
            f"Day 3: Visit Montmartre and Sacré-Cœur Basilica"
        )
    else:
        plan = (
            f"Day 1: Arrive in {dest}\n"
            f"Day 2: Explore iconic spots in {dest}\n"
            f"Day 3: Relax + sample local cuisine"
        )

    print(f"🧳 Draft itinerary:\n{plan}")
    return {**state, "itinerary": plan}


def build_enhanced_graph():
    """Build an enhanced graph with more destinations"""
    builder = StateGraph(TravelState)

    builder.add_node("destination_agent", new_destination_agent)
    builder.add_node("itinerary_agent", new_itinerary_agent)
    builder.add_node("activity_agent", activity_agent)

    builder.set_entry_point("destination_agent")
    builder.add_edge("destination_agent", "itinerary_agent")
    builder.add_edge("itinerary_agent", "activity_agent")
    builder.set_finish_point("activity_agent")

    travel_graph = builder.compile()
    print("✅ Enhanced graph compiled.\n")

    return travel_graph


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIMPLE LANGGRAPH TRAVEL AGENT")
    print("="*60)

    # Build and run simple graph
    print("\n### Part 1: Basic Graph ###")
    simple_graph = build_simple_graph()

    # Test cases
    run_example(simple_graph, "I want a relaxing beach vacation")
    run_example(simple_graph, "I want a snowy mountain retreat")

    # Build and run enhanced graph
    print("\n### Part 2: Enhanced Graph with More Destinations ###")
    enhanced_graph = build_enhanced_graph()

    run_example(enhanced_graph, "I want a city break retreat")

    print("\n" + "="*60)
    print("✨ Demo complete! Try modifying the agents or adding new ones.")
    print("="*60)
