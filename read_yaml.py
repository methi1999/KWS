import yaml
import os

#read yaml files that defines hyperparameters and the location of data
def read_yaml(path = 'config.yaml'):

	with open(path, 'r') as stream:
		try:
			with open(path) as fixed_stream:
				
				z = {**yaml.load(stream), **yaml.load(fixed_stream)}
				for path in z['dir'].values():
					if not os.path.exists(path):
						os.mkdir(path)
				return z

		except yaml.YAMLError as exc:
			return exc
