# Amazon Customer Data Analysis

## Overview

This project focuses on analyzing Amazon customer data to uncover insights about customer behavior, sentiment, and product feedback. The main tasks include:

- **Connecting to a SQLite Database**: Extracting data directly from the database.
- **Performing Sentiment Analysis**: Determining the sentiment (positive, neutral, or negative) of customer reviews.
- **Data Cleaning and Preprocessing**: Cleaning textual data by removing punctuation, stopwords, and URLs.
- **Exploratory Data Analysis (EDA)**: Gaining insights from data, including customer behaviors, review lengths, and top users.

## Features

### 1. Database Connection
- Establishes a connection with a SQLite database to extract review data.

### 2. Sentiment Analysis
- Uses the TextBlob library to compute the polarity of customer reviews.
- Polarity ranges from `-1` (negative) to `1` (positive), with `0` indicating neutral sentiment.

### 3. Text Preprocessing
- Removes punctuation and URLs.
- Eliminates stopwords for cleaner text.
- Custom functions handle edge cases in text cleaning.

### 4. Exploratory Data Analysis (EDA)
- Analyzes positive and negative reviews separately.
- Generates word clouds for both positive and negative sentiments.
- Examines the distribution of review scores.
- Highlights top users based on the number of reviews and products purchased.

### 5. User Recommendations
- Identifies top users to target with product recommendations based on their past activity.

## Requirements

The project requires the following Python libraries:

```bash
pandas
numpy
matplotlib
seaborn
sqlite3
nltk
textblob
wordcloud
plotly
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/anakail20/Amazon-Customer-Data-Analysis/tree/main
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the SQLite database file in the appropriate directory (specified in the Python script).

4. Run the script:
   ```bash
   python PythonScripts.py
   ```

## Data Description

The dataset contains the following columns:

- **Id**: Unique identifier for each review.
- **ProductId**: Unique identifier for the product.
- **UserId**: Unique identifier for the user.
- **ProfileName**: Name of the user.
- **HelpfulnessNumerator**: Number of users who found the review helpful.
- **HelpfulnessDenominator**: Number of users who evaluated the helpfulness.
- **Score**: Rating between 1 and 5.
- **Time**: Timestamp for the review.
- **Summary**: Brief summary of the review.
- **Text**: Full text of the review.

## Key Insights

- **Sentiment Analysis**: Quickly identifies whether reviews are positive, negative, or neutral.
- **Top Users**: Identifies users most likely to engage with products and leave reviews.
- **EDA Highlights**:
  - Positive reviews dominate the dataset.
  - Most reviews are concise, with the majority under 50 words.

## Visualization Examples

### Word Clouds
- Word clouds for positive and negative reviews provide visual insights into common keywords.

### Bar Charts
- Top 10 users based on product purchases.

### Box Plots
- Distribution of review lengths.

## Future Enhancements

- Incorporate advanced machine learning techniques for sentiment analysis.
- Enhance user recommendation models using collaborative filtering.
- Add deployment capabilities for real-time sentiment analysis.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- **Libraries**: Thanks to the developers of NLTK, TextBlob, and WordCloud for their amazing tools.
- **Data**: Amazon Customer Review dataset.

---

Feel free to contribute to this repository by submitting issues or pull requests.
