# RAG Demo Instructions

This script allows you to filter and rank models or datasets from a CSV file based on a natural language query. It uses a two-stage process: first, it finds the most similar items using embeddings, and then it uses a large language model to re-rank them for relevance.

## Prerequisites

Before running the script, you need to install the required Python libraries. You can do this using pip:

```bash
pip install pandas torch transformers tqdm numpy
```

You will also need to have a CSV file with either models or datasets. The CSV file must contain a column named `embeddings` which contains the pre-computed embeddings for each item. <br>
Inside the `Demo` folder, you will find a ZIP archive named `data.zip` that contains two sample CSV files:
- `models_hg_embeddings_sm.csv`: Contains embeddings for a subset of Hugging Face models.
- `datasets_hg_embeddings_sm.csv`: Contains embeddings for a subset of Hugging Face datasets.

Each file includes **2,000 items** for quick testing and demonstration purposes.

**Note**: Due to GitHub’s file size limitations, the full dataset is not included in the repository. These demo files are smaller samples of the full data.

## How to Run

You can run the script from your terminal. The basic command structure is as follows:

```bash
python rag.py <path_to_your_csv_file> "<your_query>" --mode <model_or_dataset>
```

### Arguments

*   `file_path`: (Required) The full path to the CSV file containing the data.
*   `user_prompt`: (Required) Your query in natural language, enclosed in double quotes.
*   `--mode`: (Optional) Specifies whether you are searching for a `model` or a `dataset`. Defaults to `model`.

### Examples

**Searching for a model:**

```bash
python rag.py "models_hg_embeddings_sm.csv" "What is the best model for text generation?" --mode model
```

**Searching for a dataset:**

```bash
python rag.py "datasets_hg_embeddings.csv_sm" "Which datasets are best for sentiment analysis in Italian?" --mode dataset
```

The script will then process the data and print the top-ranked, most relevant items to your console.
