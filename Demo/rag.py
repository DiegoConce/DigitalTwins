import ast
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import gc
import numpy as np
from tqdm import tqdm
import argparse

tqdm.pandas()

system_content_prompt = """
As an intelligent language model, your role is to accurately determine whether the provided data is relevant to the user's query.
Answer ONLY with 'Yes' or 'No'
"""
language_model = "meta-llama/Llama-3.1-8B-Instruct"
embedding_model = "jinaai/jina-embeddings-v3"


def convert_prompt_to_embedding(prompt):
    """Converts a text prompt into a numerical embedding using a pre-trained model.

    This function loads the `jinaai/jina-embeddings-v3` model to encode the input text.
    The resulting embedding is a vector that captures the semantic meaning of the prompt,
    suitable for similarity comparisons.

    Args:
        prompt (str): The input text to be converted into an embedding.

    Returns:
        numpy.ndarray: A NumPy array representing the embedding of the input text.
    """
    model = AutoModel.from_pretrained(embedding_model, trust_remote_code=True).to("cpu")
    embedding = model.encode(prompt, task="text-matching")
    torch.cuda.empty_cache()
    gc.collect()
    return embedding


def compute_score(embeddings, prompt):
    """Computes the similarity score between an item's embedding and a prompt embedding.

    The similarity is calculated using the dot product. A higher score indicates greater
    similarity between the item and the prompt.

    Args:
        embeddings (numpy.ndarray): The embedding vector of the item.
        prompt (numpy.ndarray): The embedding vector of the user's query.

    Returns:
        float: The resulting similarity score.
    """
    models_embedding = np.array(embeddings)
    return np.dot(models_embedding, prompt)


def filter_by_score(data, prompt, range=10):
    """Filters and ranks items in a DataFrame based on their embedding similarity to a prompt.

    This function first ensures that the 'embeddings' column is in the correct format (NumPy arrays).
    It then calculates a 'score' for each item by applying the `compute_score` function. Finally,
    it sorts the DataFrame by this score in descending order and returns the top N items.

    Args:
        data (pd.DataFrame): DataFrame containing an 'embeddings' column.
        prompt (numpy.ndarray): The prompt embedding to compare against.
        range (int, optional): The number of top items to return. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing the top N items, sorted by similarity score.
    """
    data['embeddings'] = data['embeddings'].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

    data['score'] = data['embeddings'].progress_apply(lambda x: compute_score(x, prompt))
    data.sort_values(by='score', ascending=False, inplace=True)
    return data.head(range)


def build_user_content(data, mode="model"):
    """Constructs a detailed text description for a given model or dataset item.

    This function concatenates various metadata fields from the input data row into a single
    string. This string is later used as context for the language model to determine the
    item's relevance to a user query.

    Args:
        data (pd.Series): A row from the DataFrame representing a single item.
        mode (str, optional): The type of item, either 'model' or 'dataset'. Defaults to 'model'.

    Returns:
        str: A formatted string containing the item's details.
    """
    content = ""

    if mode == "model":
        content = data['model_id'] + "\n" + data['base_model'] + "\n" + data['author'] + "\n" + data[
            'readme_file'] + "\n" + data['license'] + "\n" + data['language'] + "\n" + data['tags'] + "\n" + data[
                      'pipeline_tag'] + "\n" + data['library_name']

    elif mode == "dataset":
        content = data['dataset_id'] + "\n" + data['author'] + "\n" + data['readme_file'] + "\n" + data[
            'tags'] + "\n" + data['language'] + "\n" + data['license'] + "\n" + data['multilinguality'] + "\n" + data[
                      'size_categories'] + "\n" + data['task-categories']

    return content


def filter_by_user_prompt(data, user_prompt, mode="model"):
    """Performs a two-stage filtering of data based on a user prompt.

    First, it retrieves the most similar items using embedding scores. Second, it uses a
    large language model to re-rank and filter those items for relevance.

    Args:
        data (pd.DataFrame): The DataFrame of items to filter.
        user_prompt (str): The user's natural language query.
        mode (str, optional): The type of item, either 'model' or 'dataset'. Defaults to 'model'.

    Returns:
        pd.DataFrame: A DataFrame containing the final list of relevant items.
    """
    prompt_embedding = convert_prompt_to_embedding(user_prompt)
    data = filter_by_score(data, prompt_embedding)

    indices_to_remove = []

    tokenizer = AutoTokenizer.from_pretrained(language_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        language_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    for i, item in tqdm(data.iterrows(), total=len(data), desc="Evaluating data relevance"):
        user_content = build_user_content(item, mode)

        messages = [
            {"role": "system", "content": system_content_prompt},
            {"role": "user", "content": user_content},
        ]

        tokenized = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(tokenized, return_tensors='pt').to('cuda')
        generated_ids = model.generate(**tokenized, max_new_tokens=3000)
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        result = output[-2:]
        print(result)

        if "no" in result.lower():
            indices_to_remove.append(i)

    if indices_to_remove:
        data = data.drop(indices_to_remove)
        data = data.reset_index(drop=True)

    print(f"Shortlisted {len(data)} relevant items")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter and rank models or datasets based on a user query.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing the data.')
    parser.add_argument('user_prompt', type=str, help='The user\'s query.')
    parser.add_argument('--mode', type=str, default='model', choices=['model', 'dataset'],
                        help='The type of data to filter (model or dataset).')

    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    filtered_df = filter_by_user_prompt(df, args.user_prompt, mode=args.mode)

    print("\n--- Results ---")
    for _, row in filtered_df.iterrows():
        if args.mode == 'model':
            print(f"\n{row['model_id']} | score: {row['score']:.4f}")
            print(f"Author: {row['author']}")
            print(f"Pipeline Tag: {row['pipeline_tag']}")
        else:
            print(f"\n{row['dataset_id']} | score: {row['score']:.4f}")
            print(f"Author: {row['author']}")
            print(f"Task Categories: {row['task-categories']}")
        print("-" * 50)
