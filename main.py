import requests
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 100 if not series.empty else series

# Function to fetch news articles from NewsAPI
def fetch_news(api_key, query, from_date, end_date, page_size=100):
    url = (
        f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={end_date}"
        f"&sortBy=popularity&pageSize={page_size}&language=en&apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        raise Exception("Failed to fetch news data. Check API key and query.")

# Function to fetch stock data from Alpha Vantage
def fetch_stocks(api_key, symbol):
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&outputsize=compact&apikey={api_key}"
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
    # Calculate the daily average sentiment score
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
    else:
        print("Error: 'close' column not found in DataFrame.")
    return df

# Correlation analysis using Pearson's correlation coefficient
def analyse_correlation(df):
    correlation, p_value = pearsonr(df['daily_sentiment_score'], df['price_change'])
    print(f"Correlation: {correlation:.2f}, P-value: {p_value:.2f}")
    statement = interpret_correlation(correlation, p_value)
    print(statement)

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

# Plot a line graph comparing sentiment score and stock price over time
def plot_line_graph(df, query):
    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["normalized_sentiment"], color='blue', label='Sentiment Score', linestyle='-', linewidth=2)
    plt.plot(df["date"], df["normalized_stock_price"], color='green', label='Stock Price', linestyle='-', linewidth=2,)
    plt.title(f"Normalized Sentiment Score vs Stock Price for {query}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (0 to 100)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot a scatter plot comparing sentiment score and price change
def plot_scatter_graph(df, query):
    sns.scatterplot(x="normalized_sentiment", y="price_change", data=df, alpha=0.7, hue="normalized_sentiment", palette="coolwarm")
    plt.title(f"Sentiment Score vs Price Change for {query}")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Price Change (%)")
    plt.show()

# Main function that ties everything together
def main():
    NEWS_API = "b46d0478ffca466d8d35a7582fe8bc3e"  # Replace with your own NewsAPI key
    STOCKS_API = "A598CR21LWX8I0WH"  # Replace with your own Alpha Vantage API key
    company = input('Enter the name of the company: ')
    symbol = input('Enter the symbol(of the company entered): ')
    FROM_DATE = input("Enter a startdate (YYYY-MM-DD):[maximum of 1 month back] ")
    TO_DATE = input("Enter a end date (YYYY-MM-DD): ")
    
    # Fetch and process news data
    articles = fetch_news(NEWS_API, company, FROM_DATE, TO_DATE)
    news_df = process_data(articles)
    daily_sentiment_df = analyse_sentiment(news_df)

    # Apply rolling mean for sentiment smoothing
    news_df["smoothed_sentiment"] = news_df["sentiment_score"].rolling(window=3).mean()

    # Fetch stock data
    stock_df = fetch_stocks(STOCKS_API, symbol)

    # Merge news and stock data
    merged_df = pd.merge(daily_sentiment_df, stock_df, left_on="published_date", right_on="date", how="inner")

    # Normalize stock prices and smoothed sentiment scores
    merged_df["normalized_stock_price"] = normalize(merged_df["close"])
    merged_df["normalized_sentiment"] = normalize(merged_df["daily_sentiment_score"].fillna(0))

    # Calculate price change
    merged_df = calculate_price_change(merged_df)
    merged_df = merged_df.sort_values("date")

    # Perform correlation analysis
    analyse_correlation(merged_df)

    # Plot graphs
    plot_line_graph(merged_df, company)
    plot_scatter_graph(merged_df, company)

# Run the main function
if __name__ == "__main__":
    main()
