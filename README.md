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


The first tree parts of the workflow are already implemented in the [DataAcquisition](https://github.com/DiegoConce/DigitalTwins/blob/master/DataAcquisition.ipynb) notebook, while the last two steps are in the [Rag](https://github.com/DiegoConce/DigitalTwins/blob/master/Rag.ipynb) notebook. 
There is a [Log](https://github.com/DiegoConce/DigitalTwins/blob/master/datasets_results_log.txt) file that contains the results of the queries made by the user, which can be used to analyze the performance of the system and to improve it in the future.
