"""ClaimCheck demo.

Run:  streamlit run app.py
"""
import requests
import streamlit as st

from claimcheck.dense import QUERY_PREFIX, embed
from claimcheck.judge import judge
from claimcheck.retrieve import POOL, load_retriever

TOP_K = 3
EXAMPLES = [
    "There is an inverse relationship between hip fractures and statin use.",
    "Chenodeosycholic acid treatment increases whole-body energy expenditure.",
    "High cardiopulmonary fitness causes increased mortality rate.",
    "5% of perinatal mortality is due to low birth weight.",
]
SHOW_VERDICT = {
    "SUPPORTED": st.success,
    "REFUTED": st.error,
    "NOT_ENOUGH_INFO": st.warning,
}


@st.cache_resource
def get_retriever():
    return load_retriever()


def rank_text(ranks, row):
    return f"#{ranks[row]}" if row in ranks else f"below {POOL}"


st.title("ClaimCheck")
st.caption(
    "Checks a scientific claim against 5,183 paper abstracts, using hybrid "
    "retrieval and a language model running on this machine."
)

example = st.selectbox("Start from an example claim", EXAMPLES, index=None)
claim = st.text_input("Scientific claim", value=example or "")

if st.button("Check claim", type="primary") and claim:
    retriever = get_retriever()
    try:
        with st.spinner("Searching abstracts..."):
            vector = embed([claim], QUERY_PREFIX)[0]
            rows = retriever.hybrid(claim, vector, TOP_K)
            abstracts = [retriever.corpus[row] for row in rows]
        with st.spinner("Reading the evidence..."):
            verdict, reason = judge(claim, abstracts)
    except requests.ConnectionError:
        st.error("Could not reach Ollama. Start it with `ollama serve` and try again.")
        st.stop()

    SHOW_VERDICT[verdict](f"**{verdict.replace('_', ' ')}**: {reason}")

    # show where each abstract ranked in the two searches that were fused
    keyword_ranks = {row: rank for rank, row in enumerate(retriever.keyword(claim, POOL), 1)}
    dense_ranks = {row: rank for rank, row in enumerate(retriever.dense(vector, POOL), 1)}
    st.subheader("Abstracts the model read")
    for number, (row, abstract) in enumerate(zip(rows, abstracts), start=1):
        with st.expander(f"{number}. {abstract['title']}"):
            st.caption(
                f"Keyword rank {rank_text(keyword_ranks, row)} · "
                f"embedding rank {rank_text(dense_ranks, row)}"
            )
            st.write(abstract["text"])
