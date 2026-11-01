"""Pokémon Card Market Intelligence — Streamlit proof of concept.
Live market data requires a PkmnPrices API key and appropriate endpoint access.
"""
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

API_ROOT = 'https://api.pkmnprices.com/v1'
st.set_page_config(page_title='Pokémon Card Appraiser', page_icon='🃏', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1120px;}
h1 {font-size: clamp(1.6rem, 5vw, 2.3rem) !important;}
@media(max-width: 640px) {
 .block-container {padding-left: 0.85rem; padding-right: 0.85rem; padding-top: 0.75rem;}
 div[data-testid="stMetric"] {padding: 0.35rem 0;}
 div[data-testid="stMetricValue"] {font-size: 1.45rem;}
 h2, h3 {font-size: 1.2rem !important;}
}
</style>''', unsafe_allow_html=True)
st.title('🃏 Pokémon Card Market Intelligence')
st.caption('Phone-friendly card valuations · 30-day completed sales · supply insights')
st.info('📱 On iPhone: open the sidebar (top-left arrow) to choose Demo, CSV, or Live API. Save this webpage to your Home Screen from Safari’s Share menu.', icon=None)

@st.cache_data(ttl=900, show_spinner=False)
def get_api(path, api_key, params=None):
    res = requests.get(f'{API_ROOT}/{path.lstrip("/")}', headers={'X-API-Key': api_key}, params=params or {}, timeout=20)
    res.raise_for_status()
    return res.json()


def get_all_sales(card_id, key, graded, grader, grade, variant, limit_pages=15):
    all_rows, cursor, truncated = [], None, False
    for page in range(limit_pages):
        args = {'period': '30d', 'graded': str(graded).lower(), 'limit': 20, 'sort': 'date_desc'}
        if variant: args['variant'] = variant
        if graded and grader: args['grader'] = grader
        if graded and grade: args['grade'] = grade
        if cursor: args['cursor'] = cursor
        payload = get_api(f'cards/{card_id}/listings/ebay', key, args)
        all_rows.extend(payload.get('data') or [])
        pagination = payload.get('pagination') or {}
        cursor = pagination.get('next_cursor')
        if not pagination.get('has_more') or not cursor: break
        if page == limit_pages - 1: truncated = True
    return pd.DataFrame(all_rows), truncated


def prepare_sales(frame, variants=None):
    if frame.empty: return pd.DataFrame()
    frame = frame.copy()
    renames = {'sold_at':'date', 'listing_url':'url'}
    frame = frame.rename(columns=renames)
    if 'date' not in frame or 'price' not in frame: return pd.DataFrame()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce', utc=True).dt.tz_localize(None)
    frame['price'] = pd.to_numeric(frame['price'], errors='coerce')
    if 'currency' in frame: frame = frame[frame['currency'].astype(str).str.upper() == 'USD']
    frame = frame.dropna(subset=['date','price'])
    frame = frame[frame['price'] > 0]
    # Shared/unknown mappings cannot prove a comp belongs to a particular printing.
    if 'attribution' in frame: frame = frame[frame['attribution'].fillna('unknown') == 'exact']
    if 'id' in frame: frame = frame.drop_duplicates('id')
    return frame.sort_values('date').reset_index(drop=True)


def analysis(frame):
    day = frame.assign(day=frame['date'].dt.floor('D')).sort_values(['date', 'id'] if 'id' in frame else ['date'])
    ohlc = day.groupby('day').agg(open=('price','first'), high=('price','max'),low=('price','min'),close=('price','last'),volume=('price','size'),median=('price','median')).reset_index()
    return ohlc


def plot_market(ohlc):
    fig=make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.055, row_heights=[.76,.24])
    fig.add_trace(go.Candlestick(x=ohlc.day, open=ohlc.open, high=ohlc.high, low=ohlc.low, close=ohlc.close, name='Sold prices', increasing_line_color='#16a34a', decreasing_line_color='#dc2626'),row=1,col=1)
    fig.add_trace(go.Bar(x=ohlc.day,y=ohlc.volume,name='Observed sales', marker_color='#64748b'),row=2,col=1)
    fig.update_layout(height=430, xaxis_rangeslider_visible=False, margin=dict(l=12,r=12,t=26,b=12), template='plotly_dark', legend=dict(orientation='h',y=1.07))
    fig.update_yaxes(title_text='USD / card',row=1,col=1)
    fig.update_yaxes(title_text='Sales',row=2,col=1)
    return fig


def sample_data():
    # Entirely synthetic, illustrative transaction data; never real card prices.
    rng=np.random.default_rng(42)
    today=pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()
    rows=[]
    for offset in range(29,-1,-1):
        day=today-pd.Timedelta(days=offset)
        center=120+0.32*(29-offset)+4*np.sin((29-offset)/3.7)
        n=int(rng.integers(2,9))
        for j in range(n):
            rows.append({'id':f'demo-{offset}-{j}','date':day+pd.Timedelta(hours=int(rng.integers(8,22)),minutes=int(rng.integers(0,60))), 'price':round(max(10,rng.normal(center,5)),2), 'currency':'USD', 'attribution':'exact','title':'FICTIONAL sample card', 'url':''})
    return pd.DataFrame(rows)


def pop_display(pop):
    st.subheader('Graded population (not total cards printed)')
    st.json(pop, expanded=False)
    st.caption('Population shape varies by printing and grader; check the provider response for per-grade counts and report date.')

with st.sidebar:
    st.header('Source')
    source=st.radio('Data source',['Demonstration (synthetic)','Live API','Upload sold-sales CSV'])
    api_key=os.getenv('PKMNPRICES_API_KEY','')
    if source=='Live API':
        api_key=st.text_input('PkmnPrices API key', value=api_key, type='password', help='Key is used to call the API; not included in exported files.')
    st.markdown('### Card filters')
    graded=st.checkbox('Graded card?',value=False)
    grader=st.selectbox('Grader',['PSA','CGC','BGS','SGC']) if graded else ''
    grade=st.selectbox('Grade',['10','9.5','9','8','7','6','5','4','3','2','1']) if graded else ''
    variant=st.text_input('Printing variant (optional)',placeholder='e.g. Holofoil')
    st.caption('Do not mix raw with graded, different grades, or distinct printings in one appraisal.')

card_info={}
population=None
release_date=None
truncated=False
frame=pd.DataFrame()
if source=='Demonstration (synthetic)':
    st.warning('DEMONSTRATION ONLY: The prices and transactions below are generated examples, not real Pokémon card market data.')
    card_info={'name':'Fictional sample card', 'number':'—', 'set':{'name':'Demonstration set'}}
    frame=sample_data()
elif source=='Upload sold-sales CSV':
    up=st.file_uploader('Upload completed sales CSV',type=['csv'])
    st.caption('Required columns: sold_at (or date), price. Optional: id, currency, attribution, title, listing_url. Include only comparable card printings and grades.')
    if up is not None:
        frame=pd.read_csv(up)
        st.info('Uploaded data is user-supplied; ensure every row represents a genuine completed sale of the same variant and grade.')
else:
    if not api_key:
        st.info('Enter an API key in the sidebar to search real cards.')
    else:
        name=st.text_input('Search card name',placeholder='e.g. Charizard')
        if name:
            try:
                results=get_api('cards',api_key,{'name':name,'per_page':30}).get('data') or []
                if results:
                    selected=st.selectbox('Select exact card',range(len(results)),format_func=lambda i:f"{results[i].get('name')} · {results[i].get('number','?')} · {((results[i].get('set') or {}).get('name') or 'Unknown set')} · ID {results[i].get('id')}")
                    cid=results[selected]['id']
                    if st.button('Analyze 30-day sales',type='primary'):
                        st.session_state['selected_card']=cid
                else: st.warning('No matching cards found.')
            except requests.RequestException as e: st.error(f'Card search failed: {e}')
        cid=st.session_state.get('selected_card')
        if cid:
            try:
                card_info=get_api(f'cards/{cid}',api_key).copy()
                try:
                    set_nm=(card_info.get('set') or {}).get('name')
                    if set_nm:
                        sets=get_api('sets', api_key, {'name':set_nm, 'per_page':30}).get('data') or []
                        same=[x for x in sets if x.get('name','').casefold()==set_nm.casefold()]
                        if same: release_date=same[0].get('release_date') or same[0].get('releaseDate')
                except requests.RequestException:
                    release_date=None
                frame,truncated=get_all_sales(cid,api_key,graded,grader,grade,variant)
                try: population=get_api(f'cards/{cid}/population',api_key)
                except requests.RequestException: population=None
            except requests.RequestException as e: st.error(f'Market data unavailable: {e}')

if card_info:
    c1,c2=st.columns([1,3])
    with c1:
        if card_info.get('image_url'): st.image(card_info['image_url'],width=160)
    with c2:
        st.subheader(card_info.get('name','Card'))
        set_info=card_info.get('set') or {}
        st.write(f"Expansion / set: **{set_info.get('name','Unknown')}** · Collector number: **{card_info.get('number','Unknown')}** · Rarity: **{card_info.get('rarity','Unknown')}** · Release: **{release_date or 'Not supplied by source'}**")
        st.caption('A set name does not establish a specific booster-pack artwork. Release year is shown only if supplied by verified set metadata.')

frame=prepare_sales(frame)
if not frame.empty:
    # eBay API period filters are relative to request time; CSV/demo gets a consistent rolling cutoff.
    cutoff=pd.Timestamp.now(tz='UTC').tz_localize(None)-pd.Timedelta(days=30)
    frame=frame[frame.date>=cutoff]
if truncated: st.warning('The 30-day sold-listing results were truncated to limit API cost. Count and appraisal represent only fetched records.')
if not frame.empty:
    ohlc=analysis(frame)
    prices=frame.price.sort_values()
    low,high=prices.quantile([.25,.75])
    med=float(prices.median())
    count=len(frame)
    first, last=ohlc.iloc[0],ohlc.iloc[-1]
    change=(last['median']/first['median']-1)*100 if first['median'] else float('nan')
    a,b=st.columns(2)
    c,d=st.columns(2)
    a.metric('30-day sales',f'{count:,}')
    b.metric('Median sale',f'${med:,.2f}')
    c.metric('Middle 50% range',f'${low:,.0f}–${high:,.0f}')
    d.metric('30-day trend*',f'{change:+.1f}%')
    st.subheader('30-day completed-sale candlesticks')
    st.plotly_chart(plot_market(ohlc),use_container_width=True)
    st.caption('OHLC is ordered within each date using sale timestamps when available. When only dates are available, open and close follow source row ordering and are not true chronological market prices. High, low and daily sale counts still reflect observed records. Sales are not bid/ask order flow.')
    st.subheader('Price-point activity')
    histogram=go.Figure(go.Histogram(x=frame.price,nbinsx=18,marker_color='#7c93b8'))
    histogram.update_layout(xaxis_title='Completed sale price (USD)',yaxis_title='Number of sales', template='plotly_dark',height=285,margin=dict(l=8,r=8,t=12,b=8))
    st.plotly_chart(histogram,use_container_width=True)
    st.subheader('Comparable completed sales')
    cols=[col for col in ('date','price','title','grader','grade','variant','sale_type','url') if col in frame.columns]
    st.dataframe(frame[cols].sort_values('date',ascending=False),use_container_width=True,hide_index=True)
    st.download_button('Download filtered comps CSV',data=frame.to_csv(index=False),file_name='pokemon_comparable_sales.csv',mime='text/csv')
    st.subheader('Indicative appraisal')
    st.write(f'**${med:,.2f} median** · typical observed price range (25th–75th percentile): **${low:,.2f}–${high:,.2f}**')
    st.caption('A descriptive estimate based on filtered sales, not a formal appraisal or guaranteed selling price. With fewer than 5 comps, treat the estimate as low-confidence. Fees, shipping, condition, authenticity and exact variants matter.')
    if count<5: st.warning('Low confidence: fewer than five comparable sales.')
else:
    st.info('No matching completed-sales records to chart. A genuine appraisal requires same-card, same-printing, same-grade transactions.')
if population is not None: pop_display(population)
st.divider()
st.caption('Data sources: PkmnPrices API (live mode); CSV source if uploaded. Demonstration data is synthetic. A 30-day eBay comp count is not total global transaction volume.')
