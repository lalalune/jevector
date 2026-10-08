# Writing rules

Use ASD-STE100 Issue 9 as the writing reference.
Use short sentences and active voice. Give one instruction per sentence.
Use no more than 20 words in an instruction. Use no more than 25 words in a descriptive sentence.
Use the same term for the same function. Remove slogans, metaphors, and narrative introductions.

Code identifiers, commands, and original test inputs are data.
Do not change test inputs to change the reported results.
Define project technical terms as specified below.

This rewrite applies the writing rules. An independent check of full dictionary compliance is not available.

## Technical terms

| Term | Meaning in this project |
|---|---|
| API | Interface used to send requests to a model service |
| BM25 | Search method that calculates scores from word statistics |
| BGE-small | The `bge-small-en-v1.5` embedding model |
| Cache | Saved responses that the program can use again |
| CLS pooling | Method that uses the classification token to form a text vector |
| Configuration | Provider, model, schema, and vector size used for a test |
| Cosine similarity | Measure of the angle between two vectors |
| Dimension | One named position in a vector |
| Embedding | Numeric representation of text |
| Encoder | Process that converts an input into a vector |
| Evidence mask | Values that identify dimensions with sufficient evidence |
| Exact search | Comparison with every document vector |
| HNSW | Hierarchical Navigable Small World graph search |
| Index | Stored structure used to find matching vectors |
| Model | System that calculates answers from input text and instructions |
| Query | Request for information |
| Query-aware | Use of instructions that identify the query requirements |
| Recall@3 | Proportion of queries with the reference result in the first three positions |
| Reference document | Expected document for a test query |
| Schema | Ordered dimensions and their definitions |
| Top-1 | Correctness of the first result |
| Vector | Ordered numeric values |
| Wrong-role | Use of document instructions on a query |

Source: [ASD-STE100](https://www.asd-ste100.org/about_STE.html).
