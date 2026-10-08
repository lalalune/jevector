"""Shared documentation text. Test inputs remain unchanged."""
METHODS=[
 ('Query-aware search','Use different instructions for the query and the document. The query vector identifies required information. The document vector identifies supplied information.'),
 ('Wrong-role search','Apply the document instructions to the query. Use this test to measure the effect of encoder roles.'),
 ('BM25','Match query words to document words. Use word frequency, document frequency, and document length to calculate the score.'),
 ('BGE-small','Convert each text into a learned vector with 384 dimensions. Add the recommended search instruction to each query. Compare vectors with cosine similarity.'),
 ('HNSW','Search a graph index for nearby vectors. This method is approximate. It does not create vectors.'),
 ('Exact search','Compare the query vector with every document vector. Use this result to check HNSW.')]
METRICS=[
 ('Configuration','The provider and the number of vector dimensions.'),
 ('Query-aware top-1','The number of queries with the reference document in position 1. Use the query encoder.'),
 ('Wrong-role top-1','The number of queries with the reference document in position 1. Apply the document encoder to the query.'),
 ('HNSW/exact top-5 agreement','For each query, count the document IDs that occur in both top-five sets. Divide by five. Calculate the mean across queries.'),
 ('Top-1','The number of queries with the reference document in position 1.'),
 ('Recall@3','The proportion of queries with the reference document in the first three results.'),
 ('Cosine similarity','A score that compares vector directions. A larger score indicates a closer match. This score is not a correctness probability.')]
LABELS=['Recover access after loss of the authentication device','Reset a forgotten password','Change the account email address','Disable the additional authentication factor','Cancel future renewal','Request a refund without cancellation','Change to annual payment','Download a payment invoice','Handle HTTP 429 responses','Diagnose a request timeout','Correct HTTP 401 responses','Prevent duplicate webhook actions','Verify a webhook sender','Export records without deletion','Delete an account permanently','Restore a deleted file','Reverse accidental file deletion','Get proof of payment','Prevent duplicate purchase actions','Stop renewal after the current term']
