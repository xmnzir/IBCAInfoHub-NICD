# ================================
# IBCA Bootcamp Project
# ================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import ast

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="IBCA Information Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Site Banner
# -------------------------------
st.image("assets/ibca_banner.png", use_container_width=True)
st.markdown(
    "<h2 style='text-align:center; color:#B30000;'>Infected Blood Inquiry – IBCA Information Hub</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:16px;'>Explore FAQ, compensation, eligibility, case studies, and contact information in a visually friendly format.</p>",
    unsafe_allow_html=True
)

# ================================
# Load datasets
# ================================
summary_df = pd.read_csv("data/summary.csv")
population_df = pd.read_csv("data/population.csv")
outcomes_df = pd.read_csv("data/outcomes.csv")
comp_df = pd.read_csv("data/compensation_scheme.csv")
eligibility_df = pd.read_csv("data/eligibility.csv")
case_studies_df = pd.read_csv("data/case_studies.csv")
faqs_df = pd.read_csv("data/faqs.csv")
case_studies_df['json_awards'] = case_studies_df['json_awards'].apply(ast.literal_eval)

# ================================
# Load embedding model
# ================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_embedding_model()

@st.cache_resource
def compute_faq_embeddings(faq_list):
    return model.encode(faq_list, convert_to_tensor=True)

faq_embeddings = compute_faq_embeddings(faqs_df['question'].tolist())

# ================================
# Tabs
# ================================
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "FAQ Chatbot",
    "Basic Info",
    "Eligibility & Compensation",
    "Case Studies",
    "Contact"
])

# ================================
# TAB 0 – Overview
# ================================
with tab0:
    st.markdown("<h3 style='color:#B30000;'>About IBCA & Project Background</h3>", unsafe_allow_html=True)
    st.markdown("""
The Infected Blood Compensation Authority (IBCA) manages claims for people affected by contaminated blood and blood products.  
**Mission:** To provide fair and structured compensation to those affected and ensure transparency and support for claimants.

---

This web-based information hub and FAQ chatbot was created as a result of a **collaborative effort with colleagues from Newcastle University** working across different domains in the **National Innovation Centre for Data Spring Bootcamp**.  

The project addressed a broad topic with no straight pathway: understanding the impact of the Infected Blood scandal and the number of people affected.  

**Problem Statement:**  
Following the release of the second regulations, IBCA wanted to understand how many people were affected by the Infected Blood scandal.  

Delegates were provided with:  
- A definition of what “affected” means  
- Links to open source IBI data sources  
- Some guidance on other potential sources (ONS/NHS etc.), though delegates were encouraged to explore additional sources  

Even though we did not reach a definitive solution to the problem, the effort allowed us to **gain deep insights into the issue**, which in turn led to the creation of this platform: a **visual, interactive hub with data summaries and a semantic FAQ chatbot** to help users explore IBCA information more effectively.  
""")


# ================================
# TAB 1 – FAQ Chatbot
# ================================
with tab1:
    st.markdown("<h3 style='color:#B30000;'>FAQ Chatbot (Semantic Search)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p>Ask questions about the Infected Blood Inquiry, IBCA, compensation schemes, eligibility, or case studies. The bot will provide the most relevant answer.</p>",
        unsafe_allow_html=True
    )

    user_question = st.text_input("Type your question here:")

    if user_question:
        user_emb = model.encode(user_question, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_emb, faq_embeddings)[0]
        top_idx = int(cosine_scores.argmax())
        top_score = float(cosine_scores[top_idx])

        if top_score > 0.55:
            answer = faqs_df.iloc[top_idx]['answer']
            st.markdown("**Answer:**")
            st.success(answer)
        else:
            st.markdown("**Answer:**")
            st.warning("Sorry, no close match found. Try rephrasing your question.")

    # Display top FAQs visually
    st.markdown("<h4 style='color:#B30000;'>Top 10 Frequently Asked Questions</h4>", unsafe_allow_html=True)
    for idx, row in faqs_df.head(10).iterrows():
        with st.expander(f"{row['question']}"):
            st.markdown(f"**Answer:** {row['answer']}")

# ================================
# TAB 2 – Basic Info
# ================================
with tab2:
    st.markdown("<h3 style='color:#B30000;'>Infected Blood Inquiry Overview</h3>", unsafe_allow_html=True)
    st.markdown("""
The Infected Blood Inquiry (IBI) investigates contamination of blood and blood products with viruses such as HIV, Hepatitis B, Hepatitis C, and vCJD.  
IBCA provides structured compensation through core and supplementary routes.
""")

    st.markdown("### Summary Statistics")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### Population Breakdown")
    st.dataframe(population_df, use_container_width=True)

    st.markdown("### Mortality & Outcomes")
    st.dataframe(outcomes_df, use_container_width=True)

    st.markdown("### Visualizations")
    fig1, ax1 = plt.subplots(figsize=(9,4))
    ax1.bar(summary_df["Category"], summary_df["Value"], color="#B30000", alpha=0.85)
    ax1.set_xticks(range(len(summary_df["Category"])))
    ax1.set_xticklabels(summary_df["Category"], rotation=45, ha="right")
    ax1.set_ylabel("Number of People")
    ax1.set_title("Summary of Infected Populations")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(9,4))
    labels = population_df["Group"] + " (" + population_df["Virus"] + ")"
    ax2.bar(labels, population_df["Estimate"].fillna(0), color="#FF7F0E", alpha=0.85)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Estimated People")
    ax2.set_title("Population Estimates by Group & Virus")
    st.pyplot(fig2)

# ================================
# TAB 3 – Eligibility & Compensation
# ================================
with tab3:
    st.markdown("<h3 style='color:#B30000;'>Eligibility & Documentation</h3>", unsafe_allow_html=True)
    st.dataframe(eligibility_df, use_container_width=True)

    st.markdown("<h3 style='color:#B30000;'>Compensation Scheme</h3>", unsafe_allow_html=True)
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("<h3>IBCA Official Compensation Calculator</h3>", unsafe_allow_html=True)
    st.markdown("""
    <a href="https://ibca.org.uk/estimate-your-compensation" target="_blank">
        <button style="padding:10px 20px; font-size:16px; background-color:#B30000; color:white; border-radius:5px;">Open IBCA Calculator</button>
    </a>
    """, unsafe_allow_html=True)

# ================================
# TAB 4 – Case Studies
# ================================
with tab4:
    st.markdown("<h3 style='color:#B30000;'>Case Study Explorer</h3>", unsafe_allow_html=True)
    st.markdown("Explore example scenarios from official IBCA case-study calculations. All cases are read-only.")

    case_titles = [f"{row['id']}: {row['title']}" for idx, row in case_studies_df.iterrows()]
    selected_case = st.selectbox("Select a case study", case_titles)
    case_row = case_studies_df.loc[case_studies_df['id'] == int(selected_case.split(":")[0])].iloc[0]

    st.markdown(f"**Description:** {case_row['description']}")
    st.markdown(f"[Link to Official Case Study]({case_row['link']})")

    st.markdown("### Awards Breakdown")
    awards_df = pd.DataFrame(case_row['json_awards'].items(), columns=["Category", "Amount (£)"])
    st.dataframe(awards_df, use_container_width=True)

# ================================
# TAB 5 – Contact
# ================================
with tab5:
    st.markdown("<h3 style='color:#B30000;'>Contact IBCA</h3>", unsafe_allow_html=True)
    st.markdown("""
**General enquiries:**  
Phone: 0141 726 2397 (Mon–Fri 9am–4pm, excluding bank holidays)  
Email: ibcaenquiries@ibca.org.uk (general enquiries only; do not include personal info)

**Changes to registration information:**  
Phone: 0141 471 8886 (Mon–Fri 9am–4pm, excluding bank holidays)

**Request, delete or change information:**  
Email: ibca.datagov@ibca.org.uk

**Concerns about fraud:**  
Email: fraud@ibca.org.uk  
Report fraud: Action Fraud website or call 0300 123 2040  
In Scotland, call 101 (police)

**Community updates:**  
Sign up for newsletters or view previous updates on GOV.UK

**Media enquiries:**  
Email: ibca-media@ibca.org.uk
""")
