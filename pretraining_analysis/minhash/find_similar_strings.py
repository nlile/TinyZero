"""
Adapted from: https://github.com/eddycmu/demystify-long-cot/tree/e90cf9b30203bb74b1ea4ba008fa54bf731ad1a4

@misc{yeotong2025longcot,
      title={Demystifying Long Chain-of-Thought Reasoning in LLMs},
      author={Edward Yeo and Yuxuan Tong and Morry Niu and Graham Neubig and Xiang Yue},
      year={2025},
      eprint={2502.03373},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.03373},
}

"""

import os
import re
import time
from tqdm import tqdm
import argparse
import pandas as pd
import yaml
from multiprocessing import Pool, cpu_count
from datasketch import MinHash, MinHashLSHForest
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# --- Helper Functions ---

def sanitize_filename(s):
    """Sanitize a string to be used as a file/directory name."""
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

def get_shingles(text, shingle_size=3):
    """
    Create a set of shingles (overlapping groups of words) from the input text.
    Lower-case the text and remove punctuation.
    """
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = text.split()
    if len(tokens) < shingle_size:
        return set(tokens)
    return set(' '.join(tokens[i:i+shingle_size]) for i in range(len(tokens) - shingle_size + 1))

def build_query_index(query_categories, num_perm):
    """
    For each query string across all categories, compute its MinHash signature and build an LSH forest.
    Returns a tuple of (query_index, query_signatures, query_to_category) where:
      - query_index: a MinHashLSHForest built from all queries
      - query_signatures: a dict mapping query string to its MinHash object
      - query_to_category: a dict mapping query string to its category name
    """
    query_signatures = {}
    query_to_category = {}
    query_index = MinHashLSHForest(num_perm=num_perm)

    for category, queries in query_categories.items():
        for query in queries:
            m = MinHash(num_perm=num_perm)
            shingles = get_shingles(query)
            for shingle in shingles:
                m.update(shingle.encode("utf8"))
            query_signatures[query] = m
            query_to_category[query] = category
            query_index.add(query, m)

    query_index.index()
    return query_index, query_signatures, query_to_category

def partition_list(lst, num_partitions):
    """
    Split a list into num_partitions roughly equal parts.
    """
    k, m = divmod(len(lst), num_partitions)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(num_partitions)]

def list_input_files(input_dir, min_file=None, max_file=None):
    """
    List all parquet files in input_dir whose filename (without extension)
    falls between min_file and max_file (lexicographically). Files are sorted
    by filename (without extension).
    """
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    filtered_files = [
        f for f in all_files
        if (min_file is None or os.path.splitext(f)[0] >= min_file) and
           (max_file is None or os.path.splitext(f)[0] <= max_file)
    ]
    filtered_files = sorted(filtered_files, key=lambda f: os.path.splitext(f)[0])
    file_paths = [os.path.join(input_dir, f) for f in filtered_files]
    return file_paths

def load_queries_from_yaml(yaml_path):
    """
    Load categorized behavior queries from a YAML file.
    Expected YAML format:
      verification_queries:
        - "query string one"
      backtracking_queries:
        - "another query string"
      ...etc
    Returns a dict mapping category names to lists of queries.
    """
    # Normalize path extension if not provided
    if not os.path.exists(yaml_path):
        alt_ext = '.yml' if yaml_path.endswith('.yaml') else '.yaml'
        alt_path = yaml_path[:-len(os.path.splitext(yaml_path)[1])] + alt_ext
        if os.path.exists(alt_path):
            yaml_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find YAML file at {yaml_path} or with alternative extension {alt_ext}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Expected categories
    expected_categories = [
        "verification_queries",
        "backtracking_queries",
        "subgoal_queries",
        "backward_chaining_queries"
    ]

    # Validate categories exist
    for category in expected_categories:
        if category not in data:
            raise ValueError(f"Missing required category '{category}' in YAML file")
        if not data[category]:
            raise ValueError(f"Category '{category}' is empty in YAML file")

    return data

# --- Worker Function ---

def process_file_group(args):
    """
    Each worker process is assigned a list of file paths.
    For each file:
      - Load the parquet file
      - For each passage, compute its MinHash signature and query the query index
      - If any candidate query meets the similarity threshold, record the passage
      - Write out the matched passages to an output file with the same naming convention
    """
    (file_paths, passage_column, num_perm, query_index, query_signatures,
     query_to_category, similarity_threshold, output_dir) = args

    for file_path in file_paths:
        try:
            print(f"Process {os.getpid()} processing file {file_path}")
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if passage_column not in df.columns:
            print(f"Column {passage_column} not found in {file_path}. Skipping.")
            continue

        matched_rows = []
        # Process each passage in the file
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            passage = row[passage_column]
            sentences = sent_tokenize(passage)
            matched_by_category = {}

            for sentence in sentences:
                m_sentence = MinHash(num_perm=num_perm)
                shingles = get_shingles(sentence)
                for shingle in shingles:
                    m_sentence.update(shingle.encode("utf8"))

                candidates = query_index.query(m_sentence, len(query_signatures))
                for candidate in candidates:
                    sim = m_sentence.jaccard(query_signatures[candidate])
                    if sim >= similarity_threshold:
                        category = query_to_category[candidate]
                        if category not in matched_by_category:
                            matched_by_category[category] = set()
                        matched_by_category[category].add(candidate)

            if matched_by_category:
                row_copy = row.copy()
                # Add matches for each category
                for category, matches in matched_by_category.items():
                    row_copy[f"matched_{category}"] = ", ".join(sorted(matches))
                matched_rows.append(row_copy)

        if matched_rows:
            matched_df = pd.DataFrame(matched_rows)
            base = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, f"{base}_matched.parquet")
            matched_df.to_parquet(output_file, index=False)
            print(f"Process {os.getpid()} wrote {len(matched_df)} matched passages to {output_file}")
        else:
            print(f"Process {os.getpid()} found no matches in {file_path}")

# --- Main Processing ---

def main():
    parser = argparse.ArgumentParser(
        description="Process parquet files by partitioning files among processes. "
                    "Each process loads its group, checks passages against categorized behavior queries from YAML, "
                    "and outputs matches to a file with the same naming convention."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input parquet files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output matched parquet files.")
    parser.add_argument("--min_file", type=str, default=None,
                        help="Minimum filename (without extension, lexicographically) to process.")
    parser.add_argument("--max_file", type=str, default=None,
                        help="Maximum filename (without extension, lexicographically) to process.")
    parser.add_argument("--passage_column", type=str, default="passage",
                        help="Name of the column containing passages in the parquet files.")
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Number of permutations for MinHash.")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Minimum Jaccard similarity (approximate) to consider a passage matching a query.")
    parser.add_argument("--queries_yaml", type=str, required=True,
                        help="YAML file containing behavior queries (supports .yml or .yaml extension).")
    args = parser.parse_args()

    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load categorized queries from the YAML file
    print("Loading categorized queries from YAML...")
    query_categories = load_queries_from_yaml(args.queries_yaml)
    total_queries = sum(len(queries) for queries in query_categories.values())
    print(f"Loaded {total_queries} queries across {len(query_categories)} categories.")

    # Build the query index and signatures
    query_index, query_signatures, query_to_category = build_query_index(query_categories, args.num_perm)
    print("Query index built.")

    # List and partition the input files.
    file_paths = list_input_files(args.input_dir, args.min_file, args.max_file)
    if not file_paths:
        raise ValueError("No input files found matching the criteria.")
    print(f"Found {len(file_paths)} input files.")
    num_workers = min(cpu_count(), len(file_paths))
    partitions = partition_list(file_paths, num_workers)
    print(f"Partitioned files into {num_workers} groups.")

    # Prepare arguments for each worker.
    worker_args = []
    for part in partitions:
        worker_args.append((
            part,
            args.passage_column,
            args.num_perm,
            query_index,
            query_signatures,
            query_to_category,
            args.similarity_threshold,
            args.output_dir
        ))

    # Launch the workers.
    start_time = time.time()
    with Pool(num_workers) as pool:
        pool.map(process_file_group, worker_args)
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
