import json

from flask import Flask, make_response, request
from flask_restful import Api

from src.prediction import predict

app = Flask(__name__)
api = Api(app)


@app.route('/get_success_metrics', methods=['POST'])
def post():
    json_data = request.get_json()
    is_adult = json_data['is_adult']
    genres = json_data['genres']
    budget = json_data['budget']
    runtime = json_data['runtime']
    directors = json_data['directors']
    writers = json_data['writers']
    cast = json_data['cast']
    prod_companies = json_data['production_companies']
    output_dict = predict(
        is_adult=is_adult,
        genres=genres,
        budget=budget,
        runtime=runtime,
        directors=directors,
        writers=writers,
        cast=cast,
        prod_companies=prod_companies
    )
    response = make_response(json.dumps(output_dict, indent=2))
    response.headers['content-type'] = 'application/json'
    return response


if __name__ == '__main__':
    app.run(debug=True)
