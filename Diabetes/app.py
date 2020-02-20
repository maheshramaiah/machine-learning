import pickle;
from flask import Flask, request, jsonify, Response;

app = Flask(__name__);

with open('pickle_model.pkl', 'rb') as file:  
    pickle_model = pickle.load(file)


@app.route('/', methods=['POST'])
def isDiabetes():
	data = request.get_json();
	pred = pickle_model.predict([[data['pregnancies'], data['glucose'], data['bloodPressure'], data['skinThickness'], data['insulin'], data['bmi'], data['diabetesPedigreeFunction'], data['age']]]);

	return jsonify({
			'isDiabetes': True if pred[0] == 1 else False
		});

if __name__ == '__main__':
	app.run(debug=True)