import sys
import os
import pickle
import json


def load_model(path):
	with open(path, 'rb') as f:
		return pickle.load(f)


def parse_input(s):
	# expect JSON list or dict on command line
	try:
		return json.loads(s)
	except Exception:
		# fallback: try to parse comma-separated numbers
		try:
			return [float(x) for x in s.split(',') if x.strip()]
		except Exception:
			raise ValueError('Unable to parse input')


def predict(model, X):
	# handle single sample or batch
	import numpy as np
	arr = np.array(X)
	if arr.ndim == 1:
		arr = arr.reshape(1, -1)
	return model.predict(arr)


def main():
	if len(sys.argv) < 3:
		print('Usage: python inference.py <model.pkl> <input_json_or_csv>')
		sys.exit(2)

	model_path = sys.argv[1]
	inp = sys.argv[2]

	if not os.path.exists(model_path):
		print(f'Model file not found: {model_path}')
		sys.exit(1)

	model = load_model(model_path)
	X = parse_input(inp)
	preds = predict(model, X)
	# convert numpy types to native python for json
	try:
		print(json.dumps(preds.tolist()))
	except Exception:
		print(str(preds))


if __name__ == '__main__':
	main()

