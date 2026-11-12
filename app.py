
import hashlib
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import (
    APP_NAME,
    EXTRACTED_DATA_FILE,
    HISTORICAL_DATA_FILE,
    MAX_UPLOAD_SIZE_MB,
)
from src.rag_pipeline import ask_bill_question
from src.upload_orchestrator import (
    upload_and_index_bill,
    UploadOrchestrationError,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DATA HELPERS
# --------------------------------------------------
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_bill_records():
    """Load structured bill records for dashboard display."""
    for file_path in [EXTRACTED_DATA_FILE, HISTORICAL_DATA_FILE]:
        if Path(file_path).exists():
            with open(file_path, "r", encoding="utf-8") as file:
                records = json.load(file)

            if isinstance(records, list):
                return records, Path(file_path).name

    return [], None


def sort_bills(records):
    """Sort bills by their bill date; use month order if date is unavailable."""

    def sort_key(bill):
        bill_date = pd.to_datetime(
            bill.get("bill_date"),
            errors="coerce",
            dayfirst=True,
        )

        if not pd.isna(bill_date):
            return (0, bill_date)

        month = str(bill.get("month", "")).strip().title()

        try:
            return (1, MONTH_ORDER.index(month))
        except ValueError:
            return (2, month)

    return sorted(records, key=sort_key)


def format_currency(value):
    """Format a numeric value as Indian currency."""
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "Not available"


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "upload_success" not in st.session_state:
    st.session_state.upload_success = None

if "processed_upload_hash" not in st.session_state:
    st.session_state.processed_upload_hash = None


# --------------------------------------------------
# CUSTOM UI STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --accent: #38bdf8;
        --muted: #94a3b8;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: clamp(1.8rem, 3vw, 2.45rem);
        font-weight: 750;
        letter-spacing: -0.035em;
        line-height: 1.2;
        margin-bottom: 0.45rem;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        margin-bottom: 1.8rem;
    }

    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        margin: 0.3rem 0 1rem 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(148, 163, 184, 0.07);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 112px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.88rem;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(148, 163, 184, 0.4);
        border-radius: 12px;
        padding: 0.55rem;
    }

    div[data-testid="stForm"] {
        background: rgba(148, 163, 184, 0.045);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 1rem 1rem 0.5rem 1rem;
    }

    .source-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.5rem;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("⚡ Bill Analyzer")
    st.caption("AI-powered electricity bill assistant")
    st.divider()

    st.subheader("Upload your bill")

    uploaded_file = st.file_uploader(
        "Choose an electricity bill PDF",
        type=["pdf"],
        help=f"PDF only. Maximum size: {MAX_UPLOAD_SIZE_MB} MB.",
        key="bill_upload",
    )

    if uploaded_file is not None:
        file_size_limit = MAX_UPLOAD_SIZE_MB * 1024 * 1024

        if uploaded_file.size > file_size_limit:
            st.error(
                f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
            )
        else:
            st.success(f"Selected: {uploaded_file.name}")

            # Create a content-based identifier for this upload.
            upload_hash = hashlib.sha256(
                uploaded_file.getvalue()
            ).hexdigest()

            if (
                st.session_state.processed_upload_hash == upload_hash
            ):
                st.info("This PDF has already been processed in this session.")

            else:
                st.caption(
                    "The bill will be extracted, validated, and indexed "
                    "when you click the button below."
                )

                if st.button(
                    "Process PDF",
                    type="primary",
                    use_container_width=True,
                    key="process_bill_button",
                ):
                    try:
                        with st.spinner(
                            "Extracting bill data and updating the knowledge base..."
                        ):
                            result = upload_and_index_bill(uploaded_file)

                        st.session_state.processed_upload_hash = upload_hash

                        st.session_state.upload_success = (
                            f"Successfully processed {uploaded_file.name}. "
                            f"{result['vector']['new_chunks']} chunks added "
                            f"to the knowledge base."
                        )

                        st.rerun()

                    except UploadOrchestrationError as error:
                        st.error(str(error))

                    except Exception as error:
                        st.error(
                            "Unexpected upload error: "
                            f"{type(error).__name__}: {error}"
                        )

    if st.session_state.upload_success:
        st.success(st.session_state.upload_success)
        st.session_state.upload_success = None

    st.divider()
    st.caption("Current data source")
    st.write("Verified historical bill records")


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    f'<div class="main-title">{APP_NAME}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Understand your electricity bills, monitor '
    'monthly usage, and get answers grounded in your bill records.</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD AND SORT BILL DATA
# --------------------------------------------------
try:
    bills, data_source = load_bill_records()
    bills = sort_bills(bills)

except (OSError, json.JSONDecodeError, ValueError) as error:
    bills, data_source = [], None
    st.error(f"Unable to load bill records: {error}")


# --------------------------------------------------
# BILL OVERVIEW
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">Electricity bill overview</div>',
    unsafe_allow_html=True,
)

if bills:
    total_units = sum(
        float(bill.get("billed_units") or 0)
        for bill in bills
    )

    average_units = total_units / len(bills)
    latest_bill = bills[-1]
    latest_month = latest_bill.get("month", "Latest")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Bills available", f"{len(bills)}")
    col2.metric("Total consumption", f"{total_units:,.0f} units")
    col3.metric("Average consumption", f"{average_units:,.1f} units")

    col4.metric(
        f"{latest_month} bill amount",
        format_currency(latest_bill.get("net_bill_amount")),
    )

    st.markdown(
        f'<div class="source-note">Data loaded from '
        f'<strong>{data_source}</strong></div>',
        unsafe_allow_html=True,
    )

else:
    st.warning(
        "No structured bill records were found. Complete historical bill "
        "extraction before using the dashboard."
    )


# --------------------------------------------------
# MONTHLY CONSUMPTION CHART
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-heading">Monthly electricity consumption</div>',
    unsafe_allow_html=True,
)

if bills:
    chart_data = pd.DataFrame(
        [
            {
                "Month": str(bill.get("month", "Unknown")).title(),
                "Units consumed": float(bill.get("billed_units") or 0),
                "Bill amount": float(bill.get("net_bill_amount") or 0),
            }
            for bill in bills
        ]
    )

    month_sequence = [
        month
        for month in MONTH_ORDER
        if month in chart_data["Month"].tolist()
    ]

    chart = (
        alt.Chart(chart_data)
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
            color="#38A9E8",
        )
        .encode(
            x=alt.X(
                "Month:N",
                sort=month_sequence,
                axis=alt.Axis(
                    title="Billing month",
                    labelAngle=0,
                    labelPadding=10,
                    titlePadding=14,
                    labelFontSize=12,
                ),
            ),
            y=alt.Y(
                "Units consumed:Q",
                axis=alt.Axis(
                    title="Units consumed",
                    titlePadding=12,
                    tickMinStep=20,
                ),
            ),
            tooltip=[
                alt.Tooltip("Month:N", title="Month"),
                alt.Tooltip(
                    "Units consumed:Q",
                    title="Units",
                    format=".0f",
                ),
                alt.Tooltip(
                    "Bill amount:Q",
                    title="Bill amount",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=340)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            gridColor="#64748b",
            gridOpacity=0.18,
            domainOpacity=0.25,
        )
    )

    st.altair_chart(chart, use_container_width=True)

    with st.expander("View detailed bill records"):
        table_data = [
            {
                "Month": bill.get("month"),
                "Bill date": bill.get("bill_date"),
                "Due date": bill.get("due_date"),
                "Units": bill.get("billed_units"),
                "Net amount": format_currency(
                    bill.get("net_bill_amount")
                ),
                "Source": bill.get("source_file"),
            }
            for bill in bills
        ]

        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info("The monthly chart will appear after bill data is loaded.")


# --------------------------------------------------
# AI QUESTION AND ANSWER
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-heading">Ask your bill assistant</div>',
    unsafe_allow_html=True,
)

st.write(
    "Ask about electricity consumption, bill amounts, meter readings, "
    "or billing dates."
)

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Your question",
        placeholder="For example: Which month had the highest consumption?",
        max_chars=1000,
    )

    submitted = st.form_submit_button(
        "Ask AI",
        type="primary",
    )


if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Analyzing your electricity bills..."):
            try:
                result = ask_bill_question(question.strip())

                st.session_state.chat_history.append(
                    {
                        "question": question.strip(),
                        "answer": result.get(
                            "answer",
                            "No answer returned.",
                        ),
                        "sources": result.get("sources", []),
                        "method": result.get(
                            "retrieval_method",
                            "unknown",
                        ),
                    }
                )

            except Exception as error:
                st.error(
                    "The assistant could not process your question. "
                    f"Details: {type(error).__name__}: {error}"
                )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------
for item in reversed(st.session_state.chat_history):
    with st.container(border=True):
        st.markdown(f"**You:** {item['question']}")
        st.markdown("**Assistant:**")
        st.markdown(item["answer"])

        if item["sources"]:
            source_names = [
                f"{source.get('month', 'Unknown')} "
                f"({source.get('source_file', 'Unknown source')})"
                for source in item["sources"]
            ]

            st.caption("Sources: " + ", ".join(source_names))

        st.caption(f"Processing method: {item['method']}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "AI Electricity Bill Analyzer & Energy Assistant · "
    "Answers are grounded in the available bill records."
)
