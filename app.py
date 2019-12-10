import json

from flask import Flask, make_response, request
from flask_restful import Api

from src.prediction import predict

app = Flask(__name__)
api = Api(app)


result = None


@app.route('/post_movie_metrics', methods=['POST'])
def post():
    global result
    json_data = request.get_json()
    print(json.dumps(json_data))
    genres = json_data.get('genres', None)
    budget = json_data.get('budget', 1000000)
    runtime = json_data.get('runtime', 90)
    directors = json_data.get('directors', None)
    writers = json_data.get('writers', None)
    cast = json_data.get('cast', None)
    prod_companies = json_data.get('production_companies', None)
    output = predict(
        genres=genres,
        budget=budget,
        runtime=runtime,
        directors=directors,
        writers=writers,
        cast=cast,
        prod_companies=prod_companies
    )
    result = output
    response = make_response(json.dumps(output, indent=2))
    response.headers['content-type'] = 'application/json'
    return response


@app.route('/get_success_metrics', methods=['GET'])
def get():
    global result
    return f"{result['Revenue']} ;; {result['Rating']}"


if __name__ == '__main__':
    app.run(debug=True)
