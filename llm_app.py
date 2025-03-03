import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import openai
import matplotlib.pyplot as plt
import pandas as pd
import wikipediaapi
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re

st.title("AI Chatbot")

# Sidebar options
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_option = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo", "gpt-4"])
temperature = st.sidebar.slider("Creativity Level", 0.0, 1.0, 0.7)
tone = st.sidebar.radio("AI Personality", ["Professional", "Casual"])


def extract_relevant_text(url, query):
    """Scrapes the webpage and extracts the most relevant text based on the query."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all text from the page
        page_text = " ".join([p.get_text() for p in soup.find_all("p")])

        # Use regex to find sentences that match the query
        query_keywords = query.lower().split()[:5]  # Extract top 5 words from query
        relevant_sentences = re.findall(r"([^.]*?" + "|".join(query_keywords) + r"[^.]*\.)", page_text, re.IGNORECASE)

        if relevant_sentences:
            return " ".join(relevant_sentences[:5])  # Return the top 5 relevant sentences
        else:
            return page_text[:500]  # Fallback: First 500 characters if no direct match

    except Exception as e:
        return f"Failed to extract content: {e}"

def search_web(query):
    """Search Google, find relevant pages, and extract meaningful insights."""
    st.info("Searching Google for the best source...")
    
    results = list(search(query, num_results=5))

    if results:
        # Extract the most relevant source dynamically
        for url in results:
            if any(keyword in url for keyword in ["weather", "news", "finance", "forecast", "report", "update"]):
                extracted_text = extract_relevant_text(url, query)
                return f"**Latest Information:**\n\n{extracted_text}\n\n[Read more here]({results[0]})"

        # Fallback: If no obvious match, take first result
        extracted_text = extract_relevant_text(results[0], query)
        return f"**Latest Information:**\n\n{extracted_text}\n\n**Source:** [{results[0]}])"

    return "No search results found."

def search_wikipedia(query):
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(query)
    return page.summary if page.exists() else "No Wikipedia page found."

def generate_chart():
    data = {"Year": [2020, 2021, 2022, 2023, 2024], "Revenue": [100, 150, 200, 250, 300]}
    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    ax.plot(df["Year"], df["Revenue"], marker="o", linestyle="-")
    ax.set_title("Revenue Growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue ($M)")
    st.pyplot(fig)

def generate_image(prompt):
    response = openai.Image.create(prompt=prompt, n=1, size="512x512")
    st.image(response["data"][0]["url"])

def generate_response(input_text):
    # Keep chart and image checks below web search logic
    if "chart" in input_text.lower():
        generate_chart()
        return
    if "image" in input_text.lower():
        generate_image(input_text)
        return

    # AI Response Generation
    ai_prompt = {
        "Professional": "Answer formally.",
        "Casual": "Be fun and engaging!"
    }
    llm = ChatOpenAI(model_name=model_option, temperature=temperature, openai_api_key=openai_api_key)
    response = llm.invoke(f"{ai_prompt[tone]}\nUser: {input_text}\nAI:").content.replace("\n", " ")

    # Save chat history and display response
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append(("You", input_text))
    st.session_state.chat_history.append(("AI", response))

    st.text_area("AI Response:", response)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

text = st.text_area("Enter text:", "Ask me anything!")
# Ensure Web Search Logic Runs Before AI Generation
enable_web_search = st.sidebar.checkbox("Enable Web Search")

search_results = None
if enable_web_search:
    if "latest" in text.lower() or "current" in text.lower():
        search_results = search_web(text)
    elif "wikipedia" in text.lower():
        search_results = search_wikipedia(text.replace("wikipedia", ""))

# Keep buttons visible
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Submit", key="submit_btn"):
        if not openai_api_key.startswith("sk-"):
            st.warning("Please enter a valid OpenAI API key!")
        else:
            if enable_web_search:
                search_results = search_web(text)  # Always search first
                if search_results:
                    st.text_area("Google Search Results:", search_results, key="search_results_text", height=300)  # Unique key
                    # Store web search results in chat history
                    st.session_state.chat_history.append(("You", text))
                    st.session_state.chat_history.append(("Web Search", search_results))
                else:
                    generate_response(text)  # If no valid web search, fallback to AI
            else:
                generate_response(text)    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ensure chat history is always downloadable
chat_text = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history]) if st.session_state.chat_history else "No chat history available."
st.download_button(label="Download Chat History", data=chat_text, file_name="chat_history.txt", mime="text/plain", key="download_chat")



