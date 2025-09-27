import streamlit as st
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

st.title("📊 Dynamic Portfolio Analyzer")

# Upload JSON
uploaded_file = st.file_uploader("Upload Client Portfolio JSON", type=["json"])
if uploaded_file:
    clients_data = json.load(uploaded_file)

    client_ids = [c["clientId"] for c in clients_data]
    selected_client = st.selectbox("Select Client ID", client_ids)

    client_data = next(c for c in clients_data if c["clientId"] == selected_client)

    # -----------------------------
    # Calculate total portfolio value
    # -----------------------------
    funds = client_data["funds"]
    total_value = sum(f["amount"] for f in funds)

    # -----------------------------
    # Sector diversification
    # -----------------------------
    sector_weights = {}
    for f in funds:
        share = f["amount"]/total_value
        for sec, w in f["sectors"].items():
            sector_weights[sec] = sector_weights.get(sec,0) + w*share

    st.subheader("📌 Sector Allocation")
    st.bar_chart({k: v*100 for k,v in sector_weights.items()})

    # -----------------------------
    # Portfolio Overlap
    # -----------------------------
    def compute_overlap(f1,f2):
        h1,h2 = f1["holdings"], f2["holdings"]
        return sum(min(h1.get(s,0), h2.get(s,0)) for s in set(h1)|set(h2))*100

    overlaps=[]
    for i in range(len(funds)):
        for j in range(i+1,len(funds)):
            overlaps.append(compute_overlap(funds[i],funds[j]))
    avg_overlap = sum(overlaps)/len(overlaps) if overlaps else 0
    overlap_score = (1 - avg_overlap/100)*100

    # -----------------------------
    # HHI & Diversification Score
    # -----------------------------
    hhi = sum(v**2 for v in sector_weights.values())
    sector_score = (1 - hhi)*100
    final_score = 0.5*overlap_score + 0.5*sector_score

    if final_score > 70:
        risk_level = "Low"
    elif final_score > 50:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    st.subheader("📌 Portfolio Risk Level")
    st.write(f"Risk Level: **{risk_level}**")
    st.write(f"Diversification Score: **{final_score:.2f}**")

    # -----------------------------
    # Trader Type
    # -----------------------------
    max_sec = max(sector_weights,key=sector_weights.get)
    if max_sec in ["IT","Technology","Software"]:
        trader_type="Growth Investor"
    elif max_sec in ["Banking","Financials"]:
        trader_type="Value Investor"
    else:
        trader_type="Balanced Investor"
    st.write(f"Trader Type: **{trader_type}**")

    # -----------------------------
    # Fetch Yahoo Finance Data & Returns
    # -----------------------------
    all_stocks = list({s for f in funds for s in f["holdings"].keys()})
    ticker_map = {}
    for stock in all_stocks:
        if client_data["currency"]=="INR":
            ticker_map[stock] = stock+".NS"
        else:
            ticker_map[stock] = stock

    end = datetime.today()
    start_5y = end - timedelta(days=5*365)
    adj_close = pd.DataFrame()

    for stock,yf_ticker in ticker_map.items():
        try:
            df = yf.download(yf_ticker, start=start_5y, end=end, progress=False, auto_adjust=True)
            adj_close[stock] = df["Close"]
        except:
            st.warning(f"Skipping {yf_ticker}")

    adj_close = adj_close.dropna(axis=1,how='all')
    valid_stocks = adj_close.columns.tolist()

    weights={}
    for f in funds:
        for s,w in f["holdings"].items():
            if s in valid_stocks:
                weights[s] = weights.get(s,0) + w*f["amount"]/total_value

    if len(valid_stocks)==0:
        performance = {"oneYearReturn": None,"threeYearReturn": None,"fiveYearReturn": None}
    else:
        returns = adj_close.pct_change().dropna()
        w_array = np.array([weights[s] for s in valid_stocks])
        portfolio_daily = returns[valid_stocks].dot(w_array)
        cumulative = (1+portfolio_daily).cumprod()

        def calc_ret(days):
            if len(cumulative)<2: return None
            days=min(days,len(cumulative)-1)
            return round((cumulative.iloc[-1]/cumulative.iloc[-days]-1)*100,2)

        performance = {
            "oneYearReturn": calc_ret(252),
            "threeYearReturn": calc_ret(252*3),
            "fiveYearReturn": calc_ret(252*5)
        }

    st.subheader("📈 Portfolio Performance")
    st.write(performance)

    # -----------------------------
    # Sector Diversification Table
    # -----------------------------
    st.subheader("📊 Sector Diversification Table")
    df_sector = pd.DataFrame.from_dict({k:[v*100] for k,v in sector_weights.items()})
    st.dataframe(df_sector)

    # -----------------------------
    # Summary
    # -----------------------------
    top_sectors = sorted(sector_weights.items(), key=lambda x:x[1],reverse=True)[:2]
    st.subheader("📝 Summary")
    st.write(f"Portfolio heavily concentrated in **{top_sectors[0][0]}** and **{top_sectors[1][0]}**.")
    st.write(f"Consider diversifying into other sectors to reduce risk.")
    st.write(f"Trader Type: **{trader_type}**")
    st.write(f"Performance 1Y: {performance['oneYearReturn']}%, 3Y: {performance['threeYearReturn']}%, 5Y: {performance['fiveYearReturn']}%")

# -----------------------------
# Colab-friendly Streamlit Launch
# -----------------------------
from pyngrok import ngrok

public_url = ngrok.connect(port='8501')
print(f"Streamlit app URL: {public_url}")

!streamlit run app.py &
