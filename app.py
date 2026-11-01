import base64
import io
import json
import os
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Pokémon Card Appraiser",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container{
  max-width:980px;
  padding-top:.7rem;
  padding-left:.85rem;
  padding-right:.85rem;
  padding-bottom:4rem
}
h1{font-size:clamp(1.55rem,7vw,2.35rem)!important;margin-bottom:.2rem}
[data-testid="stMetric"]{
  border:1px solid rgba(128,128,128,.18);
  border-radius:14px;
  padding:.65rem .75rem;
  background:rgba(128,128,128,.04)
}
.scan-card{
  border:1px solid rgba(128,128,128,.2);
  border-radius:16px;
  padding:12px;
  margin:6px 0 14px 0;
}
@media(max-width:700px){
  [data-testid="column"]{min-width:48%!important;flex:1 1 48%!important}
  [data-testid="stMetricValue"]{font-size:1.35rem}
  h2,h3{font-size:1.12rem!important}
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🃏 Pokémon Card Market Intelligence")
st.caption("Scan a card, confirm the match, then analyze value, liquidity, sales activity, and graded supply.")

API_ROOT = "https://api.pkmnprices.com/v1"


def get_secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


@st.cache_data(ttl=900, show_spinner=False)
def api_get(path, key, params=None):
    r = requests.get(
        f"{API_ROOT}/{path.lstrip('/')}",
        headers={"X-API-Key": key},
        params=params or {},
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


def clean_json_text(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return text


def image_to_data_url(uploaded):
    raw = uploaded.getvalue()
    mime = getattr(uploaded, "type", None) or "image/jpeg"
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def identify_card(front, back, openai_key):
    client = OpenAI(api_key=openai_key)

    prompt = """
You are identifying a physical Pokémon trading card from photographs.

Read only what is actually visible and use visual/card-set knowledge carefully.
Return ONE JSON object and no markdown.

Required schema:
{
  "name": string|null,
  "number": string|null,
  "total_set_number": string|null,
  "set_name": string|null,
  "language": string|null,
  "edition": string|null,
  "variant": string|null,
  "rarity": string|null,
  "year": integer|null,
  "confidence": number,
  "visible_text": [string],
  "notes": string
}

Rules:
- "number" is the card's number within the set, such as "4" or "223".
- If the card visibly shows something like 4/102, return number="4" and total_set_number="102".
- set_name should be the likely official Pokémon TCG expansion, but use null if uncertain.
- edition can include 1st Edition, Unlimited, Shadowless, promo, etc. only when supported by the photo.
- variant can include Holofoil, Reverse Holofoil, Normal, etc. only when supported by the photo.
- Do NOT guess a PSA/CGC/BGS grade.
- confidence must be between 0 and 1.
- visible_text should list useful text you can actually read from the card.
"""

    content = [{"type": "input_text", "text": prompt}]
    content.append(
        {
            "type": "input_image",
            "image_url": image_to_data_url(front),
            "detail": "high",
        }
    )
    if back is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(back),
                "detail": "high",
            }
        )

    response = client.responses.create(
        model="gpt-5.5",
        input=[{"role": "user", "content": content}],
        max_output_tokens=800,
    )
    return json.loads(clean_json_text(response.output_text))


def demo_sales():
    rng = np.random.default_rng(42)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows = []
    for d in range(30):
        day = today - pd.Timedelta(days=29 - d)
        center = 120 + d * 0.35 + 4 * np.sin(d / 4)
        for j in range(int(rng.integers(2, 8))):
            rows.append(
                {
                    "id": f"demo-{d}-{j}",
                    "date": day
                    + pd.Timedelta(
                        hours=int(rng.integers(8, 22)),
                        minutes=int(rng.integers(0, 60)),
                    ),
                    "price": round(float(max(10, rng.normal(center, 5))), 2),
                    "title": "Synthetic demo sale",
                    "currency": "USD",
                }
            )
    return pd.DataFrame(rows)


def normalize_sales(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().rename(columns={"sold_at": "date", "listing_url": "url"})
    if "date" not in df.columns or "price" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df.price > 0]
    if "currency" in df.columns:
        usd = df["currency"].astype(str).str.upper().eq("USD")
        if usd.any():
            df = df[usd]
    if "attribution" in df.columns:
        exact = df["attribution"].fillna("unknown").eq("exact")
        if exact.any():
            df = df[exact]
    if "id" in df.columns:
        df = df.drop_duplicates("id")
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=30)
    return df[df.date >= cutoff].sort_values("date").reset_index(drop=True)


def ohlc(df):
    x = df.assign(day=df.date.dt.floor("D")).sort_values("date")
    return (
        x.groupby("day")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("price", "size"),
            median=("price", "median"),
        )
        .reset_index()
    )


def market_chart(x):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.76, 0.24],
    )
    fig.add_trace(
        go.Candlestick(
            x=x.day,
            open=x.open,
            high=x.high,
            low=x.low,
            close=x.close,
            name="Completed sales",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=x.day, y=x.volume, name="Sales volume"), row=2, col=1)
    fig.update_layout(
        height=430,
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=25, b=8),
        legend=dict(orientation="h"),
        template="plotly_dark",
    )
    fig.update_yaxes(title="USD", row=1, col=1)
    fig.update_yaxes(title="Sales", row=2, col=1)
    return fig


def fetch_sales(card_id, key, graded=False, grader="", grade="", variant=""):
    rows = []
    cursor = None
    for _ in range(12):
        params = {
            "period": "30d",
            "graded": str(graded).lower(),
            "limit": 20,
            "sort": "date_desc",
        }
        if grader:
            params["grader"] = grader
        if grade:
            params["grade"] = grade
        if variant:
            params["variant"] = variant
        if cursor:
            params["cursor"] = cursor
        payload = api_get(f"cards/{card_id}/listings/ebay", key, params)
        rows.extend(payload.get("data") or [])
        p = payload.get("pagination") or {}
        cursor = p.get("next_cursor")
        if not p.get("has_more") or not cursor:
            break
    return pd.DataFrame(rows)


def fetch_sales_volume(card_id, key, graded=False, variant=""):
    try:
        params = {
            "period": "30d",
            "graded": str(graded).lower(),
            "currency": "usd",
            "limit": 365,
        }
        if variant:
            params["variant"] = variant
        payload = api_get(f"cards/{card_id}/sales/volume", key, params)
        return pd.DataFrame(payload.get("data") or [])
    except Exception:
        return pd.DataFrame()


def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def normalize_num(x):
    if x is None:
        return ""
    s = str(x).strip()
    try:
        return str(int(s))
    except Exception:
        return s.lstrip("0") or "0"


def rank_candidates(candidates, ident):
    target_num = normalize_num(ident.get("number"))
    target_total = normalize_num(ident.get("total_set_number"))
    target_set = ident.get("set_name") or ""
    target_name = ident.get("name") or ""

    scored = []
    for c in candidates:
        score = 0.0
        if target_name:
            score += 3.0 * similarity(c.get("name"), target_name)
        if target_num and normalize_num(c.get("number")) == target_num:
            score += 5.0
        c_total = normalize_num(c.get("total_set_number"))
        if target_total and c_total and c_total == target_total:
            score += 4.0
        c_set = (c.get("set") or {}).get("name", "")
        if target_set:
            score += 4.0 * similarity(c_set, target_set)
        scored.append((score, c))
    scored.sort(key=lambda z: z[0], reverse=True)
    return [c for _, c in scored]


def search_candidates(ident, key):
    name = (ident.get("name") or "").strip()
    number = normalize_num(ident.get("number"))
    total = normalize_num(ident.get("total_set_number"))

    attempts = []
    if name and number and total:
        attempts.append({"name": name, "number": number, "total_set_number": total, "per_page": 25})
    if name and number:
        attempts.append({"name": name, "number": number, "per_page": 25})
    if name:
        attempts.append({"name": name, "per_page": 50})

    seen = {}
    for params in attempts:
        try:
            payload = api_get("cards", key, params)
            for c in payload.get("data") or []:
                seen[str(c.get("id"))] = c
            if len(seen) >= 8:
                break
        except Exception:
            continue

    return rank_candidates(list(seen.values()), ident)[:12]


def card_label(c):
    s = c.get("set") or {}
    n = c.get("number", "?")
    total = c.get("total_set_number")
    num_txt = f"{n}/{total}" if total else str(n)
    return f"{c.get('name','Unknown')} · {num_txt} · {s.get('name','Unknown set')}"


def render_card_summary(card, identified_year=None):
    s = card.get("set") or {}
    year = identified_year or "—"
    st.subheader(card.get("name", "Card"))
    st.write(
        f"**Set:** {s.get('name','Unknown')}  ·  "
        f"**Card #:** {card.get('number','Unknown')}"
        + (f"/{card.get('total_set_number')}" if card.get("total_set_number") else "")
        + f"  ·  **Rarity:** {card.get('rarity','Unknown')}  ·  **Year:** {year}"
    )
    if card.get("image_url"):
        st.image(card["image_url"], width=220)


with st.sidebar:
    st.header("App controls")
    source = st.radio("Data source", ["Scan / photo", "Manual search", "Demo", "Upload CSV"])
    graded = st.checkbox("Graded card", False)
    grader = st.selectbox("Grader", ["PSA", "CGC", "BGS", "SGC"]) if graded else ""
    grade = (
        st.selectbox("Grade", ["10", "9.5", "9", "8", "7", "6", "5", "4", "3", "2", "1"])
        if graded
        else ""
    )
    variant = st.text_input("Printing variant", placeholder="e.g. Holofoil")

pkmn_key = get_secret("PKMNPRICES_API_KEY")
openai_key = get_secret("OPENAI_API_KEY")

card = None
frame = pd.DataFrame()
identified_year = None

if source == "Scan / photo":
    st.header("📷 Scan a Pokémon card")
    st.write("Take a clear photo of the **front** of the card. The back is optional.")

    capture = st.camera_input("Take front photo")
    upload_front = st.file_uploader(
        "Or upload front image",
        type=["png", "jpg", "jpeg", "webp"],
        key="front_upload",
    )
    front = capture or upload_front

    back = st.file_uploader(
        "Optional: upload back image",
        type=["png", "jpg", "jpeg", "webp"],
        key="back_upload",
    )

    if front is not None:
        st.image(front, caption="Front image", use_container_width=True)

    if not openai_key:
        openai_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only to identify the card from your photo. For a permanent setup, store it in Streamlit Secrets.",
        )
    if not pkmn_key:
        pkmn_key = st.text_input(
            "PkmnPrices API key",
            type="password",
            help="Used to match the identified card to the catalog and retrieve market data.",
        )

    if front is not None and openai_key:
        if st.button("🔎 Identify card from photo", type="primary", use_container_width=True):
            with st.spinner("Reading the card..."):
                try:
                    st.session_state.identification = identify_card(front, back, openai_key)
                    st.session_state.pop("candidate_cards", None)
                    st.session_state.pop("card_id", None)
                except Exception as e:
                    st.error(f"Card identification failed: {e}")

    ident = st.session_state.get("identification")
    if ident:
        identified_year = ident.get("year")
        confidence = float(ident.get("confidence") or 0)
        st.success(
            f"Likely identification: **{ident.get('name') or 'Unknown'}**"
            + (f" · #{ident.get('number')}" if ident.get("number") else "")
            + (f" · {ident.get('set_name')}" if ident.get("set_name") else "")
        )
        st.caption(f"Vision confidence: {confidence:.0%}. Always confirm the exact card before valuing it.")

        with st.expander("What the scanner read"):
            st.json(ident)

        if pkmn_key and "candidate_cards" not in st.session_state:
            with st.spinner("Matching against the Pokémon card catalog..."):
                st.session_state.candidate_cards = search_candidates(ident, pkmn_key)

        candidates = st.session_state.get("candidate_cards") or []
        if candidates:
            st.subheader("Confirm the exact card")
            labels = [card_label(c) for c in candidates]
            choice = st.radio(
                "Select your card",
                range(len(candidates)),
                format_func=lambda i: labels[i],
                label_visibility="collapsed",
            )
            candidate = candidates[choice]
            cols = st.columns([1, 2])
            with cols[0]:
                if candidate.get("image_url"):
                    st.image(candidate["image_url"], use_container_width=True)
            with cols[1]:
                st.write(f"**{candidate.get('name','Unknown')}**")
                st.write(f"Set: {(candidate.get('set') or {}).get('name','Unknown')}")
                st.write(f"Card #: {candidate.get('number','Unknown')}")
                st.write(f"Rarity: {candidate.get('rarity','Unknown')}")

            if st.button("✅ Yes, this is my card", type="primary", use_container_width=True):
                st.session_state.card_id = candidate.get("id")
                st.rerun()
        elif pkmn_key:
            st.warning("I couldn't find a strong catalog match. Try a sharper front photo or use Manual search.")

elif source == "Manual search":
    if not pkmn_key:
        pkmn_key = st.text_input("PkmnPrices API key", type="password")
    if pkmn_key:
        q = st.text_input("Search Pokémon card", placeholder="Charizard")
        num = st.text_input("Card number (optional)", placeholder="4")
        if q:
            try:
                params = {"name": q, "per_page": 50}
                if num:
                    params["number"] = num
                results = api_get("cards", pkmn_key, params).get("data") or []
                if results:
                    idx = st.selectbox(
                        "Exact card",
                        range(len(results)),
                        format_func=lambda i: card_label(results[i]),
                    )
                    if st.button("Analyze this card", type="primary", use_container_width=True):
                        st.session_state.card_id = results[idx].get("id")
                        st.rerun()
                else:
                    st.warning("No matches found.")
            except Exception as e:
                st.error(f"Search failed: {e}")

elif source == "Demo":
    st.warning("Demo mode uses synthetic prices. It is for testing the interface only.")
    card = {
        "name": "Synthetic Charizard-style demo",
        "number": "—",
        "set": {"name": "Demo Set"},
        "rarity": "Demo only",
    }
    frame = demo_sales()

else:
    up = st.file_uploader("Upload completed-sales CSV", type=["csv"])
    st.caption("Required: date (or sold_at), price. Optional: id, title, currency, attribution, url.")
    if up is not None:
        frame = pd.read_csv(up)
        card = {
            "name": "Uploaded comparable sales",
            "number": "—",
            "set": {"name": "User supplied"},
            "rarity": "—",
        }

cid = st.session_state.get("card_id")

if cid and pkmn_key and source in ["Scan / photo", "Manual search"]:
    try:
        card = api_get(f"cards/{cid}", pkmn_key)
        render_card_summary(card, identified_year)
        current_prices = card.get("prices") or []
        if current_prices:
            with st.expander("Current market-price references"):
                st.dataframe(pd.DataFrame(current_prices), use_container_width=True, hide_index=True)

        analyze = st.button("📊 Run 30-day market appraisal", type="primary", use_container_width=True)
        if analyze or st.session_state.get("last_analyzed_card") == cid:
            st.session_state.last_analyzed_card = cid
            with st.spinner("Loading recent market data..."):
                try:
                    frame = fetch_sales(cid, pkmn_key, graded, grader, grade, variant)
                    volume_df = fetch_sales_volume(cid, pkmn_key, graded, variant)
                    st.session_state.market_frame = frame
                    st.session_state.volume_frame = volume_df
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 403:
                        st.warning(
                            "Your PkmnPrices plan does not include eBay sold comps / sales-volume data. "
                            "The free tier can still identify the card and show current catalog prices; "
                            "30-day sold comps require a plan that includes those endpoints."
                        )
                        st.session_state.market_frame = pd.DataFrame()
                        st.session_state.volume_frame = pd.DataFrame()
                    else:
                        st.error(f"Market lookup failed: {e}")
                except Exception as e:
                    st.error(f"Market lookup failed: {e}")
            frame = st.session_state.get("market_frame", pd.DataFrame())
            volume_df = st.session_state.get("volume_frame", pd.DataFrame())
    except Exception as e:
        st.error(f"Card lookup failed: {e}")

if card is not None and not cid:
    render_card_summary(card, identified_year)

frame = normalize_sales(frame)

if not frame.empty:
    x = ohlc(frame)
    q1, q3 = frame.price.quantile([0.25, 0.75])
    median = float(frame.price.median())
    first = float(x.iloc[0].median)
    last = float(x.iloc[-1].median)
    trend = ((last / first) - 1) * 100 if first else 0

    total_sales = len(frame)
    volume_df = st.session_state.get("volume_frame", pd.DataFrame())
    if volume_df is not None and not volume_df.empty and "sales" in volume_df.columns:
        try:
            total_sales = int(pd.to_numeric(volume_df["sales"], errors="coerce").fillna(0).sum())
        except Exception:
            pass

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    c1.metric("30-day observed sales", f"{total_sales:,}")
    c2.metric("Median sale", f"${median:,.2f}")
    c3.metric("Middle 50%", f"${q1:,.0f}–${q3:,.0f}")
    c4.metric("30-day trend", f"{trend:+.1f}%")

    st.subheader("Completed-sale candlesticks")
    st.plotly_chart(market_chart(x), use_container_width=True)
    st.caption(
        "Candles summarize completed transactions, not stock-style bid/ask order flow. "
        "Daily open/close depend on sale timestamps when available."
    )

    st.subheader("Price-point activity")
    hist = go.Figure(go.Histogram(x=frame.price, nbinsx=18))
    hist.update_layout(
        height=280,
        margin=dict(l=8, r=8, t=10, b=8),
        xaxis_title="Sale price (USD)",
        yaxis_title="Number of sales",
        template="plotly_dark",
    )
    st.plotly_chart(hist, use_container_width=True)

    st.subheader("Indicative appraisal")
    st.success(
        f"Estimated market center: ${median:,.2f} · "
        f"Typical observed range: ${q1:,.2f}–${q3:,.2f}"
    )
    st.caption(
        "This is a descriptive estimate from comparable completed sales, not a guaranteed selling price. "
        "Exact printing, condition, authenticity, fees and shipping matter."
    )

    cols = [
        c
        for c in ["date", "price", "title", "grader", "grade", "variant", "url"]
        if c in frame.columns
    ]
    with st.expander("Recent comparable sales"):
        st.dataframe(
            frame[cols].sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download comps CSV",
            frame.to_csv(index=False),
            "pokemon_comps.csv",
            "text/csv",
            use_container_width=True,
        )

elif cid and source in ["Scan / photo", "Manual search"]:
    st.info(
        "The card is identified. If no sold comps appear, either tap **Run 30-day market appraisal** "
        "or your market-data plan may not include the paid eBay endpoints."
    )

st.divider()
st.caption(
    "Prototype v2 · Vision identifies the likely card, but you must confirm the exact printing before appraisal. "
    "Observed sales are source-specific and are not total worldwide circulation."
)