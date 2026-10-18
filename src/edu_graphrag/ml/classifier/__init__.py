"""Subject classifier: which subject does a text fragment belong to.

`data` prepares and splits the dataset, `variants` defines the sklearn pipelines,
`evaluation` computes metrics and report tables, `model` is the MLflow pyfunc wrapper the API
serves, `train` and `register` are the CLIs.
"""
