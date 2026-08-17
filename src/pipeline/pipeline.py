import re
from src.agents.agents import create_search_agent, create_reading_agent, critic_chain, writer_chain


def strip_thinking(text: str) -> str:
    """
    Hapus blok <think>...</think> dari output model Qwen3.
    Model ini punya 'thinking mode' yang menghasilkan chain-of-thought
    panjang sebelum jawaban final. Fungsi ini membuang bagian itu.
    """
    # Hapus <think>...</think> blok (termasuk multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def pipeline_agents(topic: str) -> dict:

    state = {}

    #seacrh agent working
    print("--- STEP 1 SEARCH AGENT IS WORKING ---")

    search_agent = create_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Search for: {topic}")]
    })

    state["search_result"] = strip_thinking(search_result["messages"][-1].content)
    
    print(f"\n State Result: {state.get('search_result', '')[:300]}")

    # search reader_agent
    print("--- STEP 2 READ AGENT IS WORKING ---")
    reader_agent = create_reading_agent()
    reader_result = reader_agent.invoke({
        "messages": [
            ("user",
             f"Topic: {topic}. "
             f"Scrape the most relevant URL from these results:\n\n"
             f"{state.get('search_result', '')[:400]}")
        ]
    })

    state["scraped_content"] = strip_thinking(reader_result['messages'][-1].content)

    print(f"State scraped_content: {state.get('scraped_content', '')[:300]}")

    # writer chain
    
    print("--- STEP 3 WRITER IS DRAFTING THE REPORT ---")

    research_combined = (
        f"SEARCH RESULTS:\n{state.get('search_result', '')[:800]}\n\n"
        f"SCRAPED CONTENT:\n{state.get('scraped_content', '')[:1200]}"
    )

    state["report"] = strip_thinking(writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    }))

    print("\n Final Report\n", state.get('report', ""))

    # critic report
    print("--- STEP 4 - CRITIC IS REVIEWING THE REPORT ---")

    state["feedback"] = strip_thinking(critic_chain.invoke({
        "report": state["report"]
    }))

    print(f"Feedback Report: {state.get('feedback', '')}")

    return state