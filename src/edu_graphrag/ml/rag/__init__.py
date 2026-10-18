"""Retrieval over study-material chunks: lexical (BM25), dense (fastembed) and hybrid (RRF).

`index` builds a self-contained index directory, `model.RetrieverModel` serves it as an
MLflow pyfunc model, `experiment` evaluates variants on the QA set and registers the best.
"""
