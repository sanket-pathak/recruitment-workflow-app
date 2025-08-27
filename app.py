import os
from typing_extensions import TypedDict
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- Secrets / API key ---
# Preferred: set GOOGLE_API_KEY in your environment or Streamlit secrets.
# If you're using Streamlit Cloud, add it in "Settings > Secrets".
try:
    # If running on Streamlit Cloud/local with secrets.toml
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # st.secrets won't exist outside Streamlit

# --- LLM init (Gemini 2.0 Flash via google_genai) ---
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# --- Graph State ---
class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str

# --- Nodes ---
def categorize_experience(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as "
        "'Entry-level', 'Mid-level', or 'Senior-level'. "
        "Respond with exactly one of those phrases.\n\n"
        "Application:\n{application}"
    )
    chain = prompt | llm
    raw = chain.invoke({"application": state["application"]}).content.strip()

    # Normalize to one of the three buckets
    if "entry" in raw.lower():
        exp = "Entry-level"
    elif "mid" in raw.lower():
        exp = "Mid-level"
    elif "senior" in raw.lower():
        exp = "Senior-level"
    else:
        exp = "Unknown"
    return {"experience_level": exp}

def assess_skillset(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "You are screening applications for a C++ developer role. "
        "Based on the application below, respond with exactly either 'Match' or 'No Match'. "
        "Do not explain.\n\n"
        "Application:\n{application}"
    )
    chain = prompt | llm
    raw = chain.invoke({"application": state["application"]}).content.strip()

    low = raw.lower()
    if "no match" in low:
        skill = "No Match"
    elif "match" in low:
        skill = "Match"
    else:
        # Default conservative choice
        skill = "No Match"
    return {"skill_match": skill}

def schedule_interview(state: State) -> State:
    return {"response": "Interview Scheduled"}

def escalate_to_recruiter(state: State) -> State:
    return {"response": "Candidate Escalated"}

def reject_application(state: State) -> State:
    return {"response": "Candidate Rejected"}

# --- Build Graph ---
workflow = StateGraph(State)
workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skillset)
workflow.add_node("schedule_interview", schedule_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)

def route_app(state: State) -> str:
    # If skills match, schedule directly.
    if state.get("skill_match") == "Match":
        return "schedule_interview"
    # Otherwise, senior candidates go to recruiter.
    if state.get("experience_level") == "Senior-level":
        return "escalate_to_recruiter"
    # Others are rejected.
    return "reject_application"

workflow.add_edge(START, "categorize_experience")
workflow.add_edge("categorize_experience", "assess_skillset")
workflow.add_conditional_edges(
    "assess_skillset",
    route_app,
    {
        "schedule_interview": "schedule_interview",
        "escalate_to_recruiter": "escalate_to_recruiter",
        "reject_application": "reject_application",
    },
)
workflow.add_edge("schedule_interview", END)
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)

app = workflow.compile()

def run_candidate_screening(application: str):
    results = app.invoke({"application": application})
    return {
        "experience_level": results.get("experience_level", "Unknown"),
        "skill_match": results.get("skill_match", "No Match"),
        "response": results.get("response", "Candidate Rejected"),
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Recruitment Workflow", page_icon="✅", layout="centered")
st.title("Recruitment Agency Workflow")

st.caption("Model: Gemini 2.0 Flash via LangChain + LangGraph")

with st.form("screen_form"):
    application_text = st.text_area(
        "Paste the candidate application here:",
        value="I have 4 years of experience in C++ and STL, worked on embedded systems and performance optimization.",
        height=180,
    )
    submitted = st.form_submit_button("Run Screening")

if submitted:
    with st.spinner("Processing…"):
        results = run_candidate_screening(application_text)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Experience Level", results["experience_level"])
    c2.metric("Skill Match", results["skill_match"])
    c3.metric("Action", results["response"])

    with c3:
        st.markdown(
            f"""
            <div style="font-size:14px; font-weight:bold; color:#333;">
                {results["response"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.code(application_text, language="markdown")

st.sidebar.header("Configuration")
st.sidebar.write(
    "Set `GOOGLE_API_KEY` as an environment variable or via Streamlit secrets."
)
