# Digital Twins: Retrieval-Augmented Generation for Hugging Face Models
## Introduction
The goal of this project is to build a lightweight Retrieval-Augmented Generation (RAG) system over a dataset of information about Large Language Models.
This system allows users to search and retrieve the most relevant models based on natural language queries by leveraging text embeddings and metadata filtering. 
The approach is designed to support use cases such as model discovery, recommendation, and ranking without relying on extensive computational resources. 

## Workflow
The workflow consists of serveral steps:
1. **Data Collection**: Gather metadata about Hugging Face models, including model IDs, authors, pipeline tags, and embeddings.
2. **Data Preprocessing**: Clean and preprocess the collected data, including converting embeddings to a suitable format.
3. **Embedding Generation**: Use a pre-trained model to generate embeddings for the collected metadata.
4. **Query Processing**: Implement a query processing system that converts user queries into embeddings and retrieves relevant models based on similarity scores.
5. **Ranking and Filtering**: Rank and filter the retrieved models based on their relevance to the user query, considering additional metadata such as downloads, likes, and publication date.


The first tree parts of the workflow are already implemented in the [DatasetAcquisition](https://github.com/DiegoConce/DigitalTwins/blob/master/DataAcquisition.ipynb) notebook and [ModelsAcquisition](https://github.com/DiegoConce/DigitalTwins/blob/master/ModelsAcquisition.ipynb) Notebook, while the last two steps are in the [Rag](https://github.com/DiegoConce/DigitalTwins/blob/master/Rag.ipynb) notebook. 

---

### Results Directory

The [Results](https://github.com/DiegoConce/DigitalTwins/tree/master/Results) directory contains the output of the RAG system, including:

- `datasets_results.txt` – RAG results for Hugging Face dataset metadata using standard (unweighted) embeddings.
- `datasets_results_weighted_emb.txt` – RAG results for Hugging Face dataset metadata using **weighted** embeddings.
- `models_results.txt` – RAG results for Hugging Face model metadata using standard (unweighted) embeddings.
- `models_results_weighted_emb.txt` – RAG results for Hugging Face model metadata using **weighted** embeddings.

---

### Demo Directory
The [Demo](https://github.com/DiegoConce/DigitalTwins/tree/master/Demo) directory contains an example script that allows you to filter and rank models or datasets from a CSV file based on a natural language query.<br>
It uses a two-stage process: first, it finds the most similar items using embeddings, and then it uses a large language model to re-rank them for relevance.<br>
The script is designed to be run from the command line and requires a CSV file with pre-computed embeddings for each item. The demo includes sample CSV files for both models and datasets, which can be used for quick testing and demonstration purposes.






