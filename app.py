import streamlit as st
from langchain_core.messages import HumanMessage
from agent_engine import agent_app

# Set up clean web browser configurations
st.set_page_config(page_title="Smart Financial Portfolio Analyst", page_icon="📊", layout="wide")

st.title("📊 Smart Financial Portfolio Analyst")
st.subheader("Autonomous Cyclical Agent for Real-time Compliance Checking")
st.markdown("---")

# Main user query interface
user_query = st.text_input(
    "Enter your portfolio action or allocation adjustment query:",
    placeholder="e.g., I want to invest 30% of my total portfolio capital directly into NVIDIA (NVDA) stock."
)

if user_query:
    st.markdown("### 🕸️ Live Multi-Node Graph Trace")
    
    # Pack the user input into the initial LangGraph State payload
    initial_inputs = {
        "messages": [HumanMessage(content=user_query)], 
        "risk_assessment_attempts": 0, 
        "compliance_approved": False
    }
    
    # We use st.status to dynamically show the user exactly which node is executing in real-time
    with st.status("Initializing Graph Workspace Engine...", expanded=True) as status:
        final_state = None
        
        # Stream the nodes of the graph as they fire sequentially
        for chunk in agent_app.stream(initial_inputs, stream_mode="updates"):
            for node_name, state_payload in chunk.items():
                st.write(f"🔄 **Current State Node Execution:** `{node_name}`")
                
                # Contextually log the status information based on the executing node
                if node_name == "analyzer":
                    st.caption("Central Reasoning engine calculating user intent and picking apart target equity tickers...")
                elif node_name == "tools":
                    st.caption("Scraping live stock metrics from yfinance and pulling active rule boundaries from the FAISS Vector Database...")
                elif node_name == "evaluator":
                    st.caption("Cross-examining real-time math against compliance manuals to run risk calculations...")
                    
                    # If it rejected, show the temporary failure notice inside the loop expander
                    if "messages" in state_payload and not state_payload.get("compliance_approved", True):
                        st.warning(f"⚠️ Guardrail violation captured! Triggering auto-correction recovery loop...")
                        
                final_state = state_payload

        status.update(label="Graph Pipeline Execution Complete!", state="complete", expanded=False)
        
    st.markdown("---")
    st.markdown("### 📋 Final Executive Verdict Report")
    
    # Grab the final report message appended right before the graph ended execution
    if final_state and "messages" in final_state:
        st.markdown(final_state["messages"][-1].content)
    else:
        st.info("System compiled completely but did not return standard message sequences. Adjust state payloads.")
