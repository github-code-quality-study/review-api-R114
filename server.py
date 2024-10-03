import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')
valid_locations = {
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona"
}
class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(reviews, indent=2).encode("utf-8")
            filtered_reviews = reviews

            # Extract query parameters from the query string
            query_string = parse_qs(environ["QUERY_STRING"])
            
            # Filter reviews based on query parameters
            if query_string:
                if 'location' in query_string:
                    filtered_reviews = [review for review in filtered_reviews if review['Location'] == query_string['location'][0]]
            
                if 'start_date' in query_string:
                    start_date = datetime.strptime(query_string['start_date'][0], '%Y-%m-%d')
                    filtered_reviews = [review for review in filtered_reviews if start_date <= datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S')]

                if 'end_date' in query_string:
                    end_date = datetime.strptime(query_string['end_date'][0], '%Y-%m-%d')
                    filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]
            # Analyze sentiment for each review
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            # Sort reviews based on sentiment score
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)
            
            # Create the final response body
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            
            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Read the request body
            request_body_size = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(request_body_size).decode("utf-8")
            data = parse_qs(request_body)
            # print(data)
            try:
                location = data["Location"][0]
                body = data["ReviewBody"][0]
            except KeyError:
                location = None
                body = None
            if location and body and location in valid_locations:
                # print(f"Received a new review for {location}-{body}")
                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "Location": location,
                    "ReviewBody": body,
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                reviews.append(new_review)
                response = json.dumps(new_review, indent=2).encode("utf-8")
                status = "201 Created"
            else:
                error_message = "Location and ReviewBody are required fields"
                response = json.dumps({"error": error_message}, indent=2).encode("utf-8")
                status = "400 Bad Request"
            
            # Set the appropriate response headers
            start_response(status, [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response)))
            ])
            # print(status)
            return [response]
            

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
