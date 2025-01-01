import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import datetime
import fpdf
from fpdf import FPDF

# Hardcoded API keys
NEWS_API_KEY = "b46d0478ffca466d8d35a7582fe8bc3e"
ALPHA_VANTAGE_API_KEY = "A598CR21LWX8I0WH"

# Define normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 100 if not series.empty else series

# Function to fetch news articles from NewsAPI
def fetch_news(query, from_date, end_date, page_size=100):
    url = (
        f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={end_date}"
        f"&sortBy=popularity&pageSize={page_size}&language=en&apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        raise Exception("Failed to fetch news data. Check query or API key.")

# Function to fetch stock data from Alpha Vantage
def fetch_stocks(symbol):
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        if "Time Series (Daily)" in json_data:
            data = json_data["Time Series (Daily)"]
            df = pd.DataFrame(data).T
            df.columns = ["open", "high", "low", "close", "volume"]
            df = df[["close", "volume"]].reset_index().rename(columns={"index": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = df["close"].astype(float)
            return df.sort_values("date")
        else:
            raise Exception(f"Unexpected API response: {json_data.get('Error Message', json_data)}")
    else:
        raise Exception(f"Failed to fetch stock data. Status code: {response.status_code}")

# Function to process news articles into a DataFrame
def process_data(news_article):
    data = []
    for article in news_article:
        data.append({
            "headline": article["title"],
            "source": article['source']['name'],
            "published_date": article["publishedAt"][:10]
        })
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["headline"], keep="first")
    df['published_date'] = pd.to_datetime(df['published_date'])
    return df

# Sentiment analysis function to score the sentiment of headlines
def senti_score(headline):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(headline)['compound'] * 100

# Function to apply sentiment analysis to the news DataFrame
def analyse_sentiment(news_df):
    news_df["sentiment_score"] = news_df["headline"].apply(senti_score)
    daily_sentiment = news_df.groupby("published_date").agg(
        daily_sentiment_score=pd.NamedAgg(column="sentiment_score", aggfunc="mean")
    ).reset_index()
    return daily_sentiment

# Merge news and stock data based on the published date and stock data date
def merge_data(news_df, stock_df):
    return pd.merge(news_df, stock_df, right_on='date', left_on="published_date", how='inner')

# Calculate price change percentage for stock data
def calculate_price_change(df):
    if 'close' in df.columns:
        df['price_change'] = df['close'].pct_change() * 100  # Calculate percentage change
        df['price_change'] = df['price_change'].fillna(0)  # Fill NaN with 0 for the first row
    return df

# Correlation analysis using Pearson's correlation coefficient
def interpret_correlation(correlation, p_value):
    if abs(correlation) >= 0.3:
        correlation_strength = "strong"
    elif abs(correlation) >= 0.10:
        correlation_strength = "moderate"
    else:
        correlation_strength = "weak"

    if p_value < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"

    return f"The correlation between sentiment scores and stock prices is {correlation_strength}, and it is {significance}."

# Define plotting functions
def plot_line_graph(df, company):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["date"], df["normalized_sentiment"], color='blue', label='Sentiment Score', linestyle='-', linewidth=2)
    ax.plot(df["date"], df["normalized_stock_price"], color='green', label='Stock Price', linestyle='-', linewidth=2)
    ax.set_title(f"Normalized Sentiment Score vs Stock Price for {company}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value (0 to 100)")
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_scatter_graph(df, company):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="normalized_sentiment", y="price_change", data=df, alpha=0.7, hue="normalized_sentiment", palette="coolwarm", ax=ax)
    ax.set_title(f"Sentiment Score vs Price Change for {company}")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Price Change (%)")
    st.pyplot(fig)

# Function to generate PDF report
def generate_pdf(merged_df, company):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    title = f"Sentiment and Stock Analysis Report for {company}"
    pdf.cell(200, 10, txt=title, ln=True, align='C')

    # Add Sentiment and Stock Analysis Data
    pdf.ln(10)
    pdf.cell(200, 10, txt="Analysis Summary", ln=True, align='L')

    # Adding tables for sentiment and stock data
    pdf.set_font("Arial", size=10)
    pdf.ln(5)
    pdf.cell(100, 10, txt="Date", border=1, align='C')
    pdf.cell(50, 10, txt="Sentiment Score", border=1, align='C')
    pdf.cell(50, 10, txt="Stock Price Change (%)", border=1, align='C')
    pdf.ln(10)

    for index, row in merged_df.iterrows():
        pdf.cell(100, 10, txt=row['date'].strftime('%Y-%m-%d'), border=1)
        pdf.cell(50, 10, txt=f"{row['normalized_sentiment']:.2f}", border=1)
        pdf.cell(50, 10, txt=f"{row['price_change']:.2f}", border=1)
        pdf.ln(10)

    pdf_file = f"{company}_Sentiment_Stock_Analysis.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Streamlit App
st.title("Sentiment and Stock Analysis Tool")

# Main Screen Inputs
st.header("Inputs")
st.write("Provide the required details below.")

col1, col2 = st.columns(2)

with col1:
    company = st.text_input("Company Name", "Tesla", help="Enter the company name you want to analyze.")
    symbol = st.text_input("Company Symbol", "TSLA", help="Enter the stock symbol for the company.")

with col2:
    today = datetime.date.today()
    default_start_date = today - datetime.timedelta(days=30)
    from_date = st.date_input("From Date", value=default_start_date, help="Select the start date for analysis.")
    to_date = st.date_input("To Date", value=today, help="Select the end date for analysis.")

if st.button("Run Analysis"):
    try:
        st.write(f"Analyzing data for {company} ({symbol}) from {from_date} to {to_date}.")
        
        # Fetch and process news
        articles = fetch_news(company, str(from_date), str(to_date))
        news_df = process_data(articles)
        daily_sentiment_df = analyse_sentiment(news_df)

        # Fetch stock data
        stock_df = fetch_stocks(symbol)

        # Merge and process data
        merged_df = merge_data(daily_sentiment_df, stock_df)
        merged_df["normalized_stock_price"] = normalize(merged_df["close"])
        merged_df["normalized_sentiment"] = normalize(merged_df["daily_sentiment_score"].fillna(0))
        merged_df = calculate_price_change(merged_df)

        # Calculate correlation
        correlation, p_value = pearsonr(merged_df['normalized_sentiment'], merged_df['price_change'])
        statement = interpret_correlation(correlation, p_value)

        # Display correlation results
        st.write("### Correlation Results")
        st.write(f"**Correlation:** {correlation:.2f}")
        st.write(f"**P-value:** {p_value:.2f}")
        st.markdown(statement)

        # Display graphs
        st.write("### Line Graph")
        plot_line_graph(merged_df, company)

        st.write("### Scatter Plot")
        plot_scatter_graph(merged_df, company)

        # PDF Generation
        st.write("### Download PDF Report")
        pdf_file = generate_pdf(merged_df, company)
        st.download_button("Download PDF Report", data=open(pdf_file, "rb"), file_name=pdf_file, mime="application/pdf")

    except Exception as e:
        st.error(f"An error occurred: {e}")
