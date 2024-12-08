📊 Stock Sentiment Analysis </br>
This project explores the relationship between public sentiment and stock price movements by analyzing real-time news and stock data. The goal is to identify trends and patterns in how sentiment affects stock prices using Python and data visualization techniques.

🔍 Overview
- Fetches news articles about a company using NewsAPI.
- Performs sentiment analysis on headlines with NLTK's SentimentIntensityAnalyzer.
- Retrieves stock price data using Alpha Vantage API.
- Calculates daily price changes and correlates them with sentiment scores.
- Generates insightful visualizations:
  - Line graph comparing normalized sentiment scores and stock prices.
  - Scatter plot analyzing the correlation between sentiment and price changes.
    
🛠 Features
- Real-Time Data Retrieval: Fetches the latest news and stock data based on user input.
- Sentiment Analysis: Assigns a sentiment score (positive/neutral/negative) to each headline.
- Data Normalization: Normalizes stock prices and sentiment scores for better comparison.
- Correlation Analysis: Evaluates the statistical relationship between sentiment and stock price changes.
- Visualization: Interactive plots for better understanding of trends and correlations.

🛠 Technologies Used
Python: Pandas, Seaborn, Matplotlib, NLTK, Requests </br>
APIs: NewsAPI, Alpha Vantage </br>
Libraries: SentimentIntensityAnalyzer (NLP) </br>
