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
import psutil
from datetime import datetime
from tqdm import tqdm
import argparse
import pandas as pd
import yaml
import json
from collections import defaultdict, Counter
import datasets
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager, Lock
from datasketch import MinHash, MinHashLSHForest
import nltk
from nltk.tokenize import sent_tokenize
from loguru import logger
import pyarrow.parquet as pq

try:
    nltk.download('punkt_tab')
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    logger.warning("Will attempt to continue but sentence tokenization may fail")


def setup_nltk():
    """Setup NLTK data in a safe way."""
    try:
        nltk.data.find('punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            logger.warning("Will attempt to continue but sentence tokenization may fail")

def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)

def get_memory_usage_gb() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)

def ensure_dataset_cached(dataset_name: str, cache_dir: str, max_size_gb: float = 10.0) -> list[str]:
    """
    Ensure dataset is cached locally, downloading if needed.
    Returns list of parquet file paths.

    Args:
        dataset_name: Name of dataset on HuggingFace
        cache_dir: Directory to cache dataset
        max_size_gb: Maximum allowed size in GB for the dataset
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists() or not any(cache_path.glob("*.parquet")):
        logger.info(f"No cached dataset found at {cache_dir}, downloading {dataset_name}...")
        os.makedirs(cache_dir, exist_ok=True)

        # Download with streaming to check size
        ds = datasets.load_dataset(dataset_name, split='train', streaming=True)

        # Save in chunks of 100k examples
        chunk_size = 100_000
        current_chunk = 0
        current_size_gb = 0
        chunk_files = []

        iterator = iter(ds)
        current_batch = []

        for example in tqdm(iterator, desc="Downloading dataset"):
            current_batch.append(example)
            if len(current_batch) >= chunk_size:
                output_file = cache_path / f"chunk_{current_chunk:03d}.parquet"
                chunk_ds = datasets.Dataset.from_list(current_batch)
                chunk_ds.to_parquet(str(output_file))

                chunk_size_gb = get_file_size_gb(str(output_file))
                current_size_gb += chunk_size_gb

                if current_size_gb > max_size_gb:
                    logger.warning(f"Dataset exceeds {max_size_gb}GB limit! Truncating to current chunks.")
                    chunk_files.append(str(output_file))
                    break

                chunk_files.append(str(output_file))
                current_batch = []
                current_chunk += 1

        # Save any remaining examples and update size
        if current_batch:
            output_file = cache_path / f"chunk_{current_chunk:03d}.parquet"
            chunk_ds = datasets.Dataset.from_list(current_batch)
            chunk_ds.to_parquet(str(output_file))
            chunk_size_gb = get_file_size_gb(str(output_file))
            current_size_gb += chunk_size_gb
            chunk_files.append(str(output_file))

        logger.info(f"Cached dataset in {len(chunk_files)} chunks, total size: {current_size_gb:.2f}GB")
        return chunk_files
    else:
        # Check existing files
        parquet_files = list(cache_path.glob("*.parquet"))
        total_size_gb = sum(get_file_size_gb(str(f)) for f in parquet_files)
        if total_size_gb > max_size_gb:
            logger.warning(f"Cached dataset is {total_size_gb:.2f}GB, exceeding {max_size_gb}GB limit!")
        return [str(f) for f in parquet_files]

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
        "backward_chaining_queries",
        "clarify_queries",
        "pivot_queries"
    ]

    # Validate categories exist
    for category in expected_categories:
        if category not in data:
            raise ValueError(f"Missing required category '{category}' in YAML file")
        if not data[category]:
            raise ValueError(f"Category '{category}' is empty in YAML file")

    return data

# --- Worker Function ---

def get_checkpoint_file(output_dir: Path) -> Path:
    """Get path to checkpoint file."""
    return output_dir / "checkpoint.yaml"

def save_checkpoint(output_dir: Path, completed_files: list[str]):
    """Save checkpoint of completed files."""
    checkpoint_file = get_checkpoint_file(output_dir)
    with open(checkpoint_file, 'w') as f:
        yaml.dump({'completed_files': completed_files}, f)

def load_checkpoint(output_dir: Path) -> list[str]:
    """Load checkpoint of completed files."""
    checkpoint_file = get_checkpoint_file(output_dir)
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = yaml.safe_load(f)
            return checkpoint.get('completed_files', [])
    return []

class StatsTracker:
    """Track and update statistics for matching results."""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.stats_file = output_dir / 'running_stats.json'
        self.stats = {
            'total_rows_processed': 0,
            'total_rows_matched': 0,
            'matches_by_category': defaultdict(int),
            'category_percentages': {},
            'last_update': None
        }
        self.load_existing_stats()

    def load_existing_stats(self):
        """Load existing stats if they exist."""
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)

    def update(self, new_rows: int, matches_by_category: dict):
        """Update stats with new batch of processed rows."""
        self.stats['total_rows_processed'] += new_rows

        # Update category matches
        for category, count in matches_by_category.items():
            self.stats['matches_by_category'][category] += count

        # Update total matches
        self.stats['total_rows_matched'] = sum(self.stats['matches_by_category'].values())

        # Update percentages
        if self.stats['total_rows_processed'] > 0:
            total = self.stats['total_rows_processed']
            self.stats['category_percentages'] = {
                cat: (count / total) * 100
                for cat, count in self.stats['matches_by_category'].items()
            }
            self.stats['overall_match_percentage'] = (self.stats['total_rows_matched'] / total) * 100

        self.stats['last_update'] = datetime.now().isoformat()
        self.save()
        self.display()

    def save(self):
        """Save current stats to file."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def display(self):
        """Display current statistics."""
        logger.info("\n=== Current Processing Statistics ===")
        logger.info(f"Total rows processed: {self.stats['total_rows_processed']:,}")
        logger.info(f"Total rows with matches: {self.stats['total_rows_matched']:,}")
        logger.info("\nMatches by category:")
        for cat, count in self.stats['matches_by_category'].items():
            percentage = self.stats['category_percentages'].get(cat, 0)
            logger.info(f"  {cat}: {count:,} ({percentage:.2f}%)")
        logger.info(f"\nOverall match rate: {self.stats.get('overall_match_percentage', 0):.2f}%")
        logger.info("=====================================")

def process_file_group(args):
    """
    Each worker process is assigned a list of file paths.
    For each file:
      - Load the parquet file in chunks using pyarrow.iter_batches
      - For each passage, compute its MinHash signature and query the query index
      - If any candidate query meets the similarity threshold, record the passage
      - Write out the matched passages to an output file with the same naming convention
    """
    (file_paths, passage_column, num_perm, query_index, query_signatures,
     query_to_category, similarity_threshold, output_dir, stats_tracker,
     shared_completed, checkpoint_lock) = args

    chunk_size = 10000  # Process 10k rows at a time
    mem_limit_gb = 4.0  # Warn if process exceeds 4GB

    for file_path in file_paths:
        try:
            print(f"Process {os.getpid()} processing file {file_path}")

            # Use pyarrow to read parquet file in batches
            parquet_file = pq.ParquetFile(file_path)
            # Optional: get total rows from metadata (if needed) for progress display
            total_rows = parquet_file.metadata.num_rows

            matched_rows = []
            chunk_stats = defaultdict(int)
            rows_processed = 0
            chunk_num = 0

            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas(split_blocks=True, self_destruct=True)
                if passage_column not in chunk_df.columns:
                    print(f"Column {passage_column} not found in {file_path}. Skipping.")
                    break

                rows_in_chunk = len(chunk_df)
                rows_processed += rows_in_chunk
                chunk_matches = defaultdict(int)

                # Process each passage in the chunk using itertuples for efficiency
                for row in tqdm(chunk_df.itertuples(index=False), total=len(chunk_df), desc=f"Chunk {chunk_num}"):
                    # Check memory usage
                    mem_usage = get_memory_usage_gb()
                    if mem_usage > mem_limit_gb:
                        logger.warning(f"High memory usage: {mem_usage:.2f}GB")

                    # Access the passage value safely via getattr
                    passage = getattr(row, passage_column)
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
                                    chunk_matches[category] += 1
                                matched_by_category[category].add(candidate)

                    if matched_by_category:
                        row_copy = chunk_df.iloc[chunk_df.index.get_loc(row.Index)].copy() if hasattr(row, 'Index') else row
                        # If row is a namedtuple, convert to dict via _asdict() if needed
                        # Append new keys for each matching category
                        row_dict = row._asdict() if hasattr(row, '_asdict') else row.__dict__
                        for category, matches in matched_by_category.items():
                            row_dict[f"matched_{category}"] = ", ".join(sorted(matches))
                        matched_rows.append(row_dict)

                # Update stats after processing each chunk
                stats_tracker.update(rows_in_chunk, dict(chunk_matches))

                # Save intermediate results every 50k matched rows
                if len(matched_rows) >= 50000:
                    matched_df = pd.DataFrame(matched_rows)
                    base = os.path.splitext(os.path.basename(file_path))[0]
                    output_file = os.path.join(output_dir, f"{base}_matched_part{chunk_num:03d}.parquet")
                    matched_df.to_parquet(output_file, index=False)
                    print(f"Process {os.getpid()} wrote {len(matched_df)} matched passages to {output_file}")
                    matched_rows = []
                chunk_num += 1

            # Save any remaining matches
            if matched_rows:
                matched_df = pd.DataFrame(matched_rows)
                base = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(output_dir, f"{base}_matched_final.parquet")
                matched_df.to_parquet(output_file, index=False)
                print(f"Process {os.getpid()} wrote final {len(matched_df)} matched passages to {output_file}")

            # Update shared checkpoint in a multiprocessing-safe manner
            with checkpoint_lock:
                if file_path not in shared_completed:
                    shared_completed.append(file_path)
                    save_checkpoint(Path(output_dir), list(shared_completed))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

# --- Main Processing ---

def get_default_output_dir() -> Path:
    """Get default output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(__file__).parent.parent / 'data' / 'results' /
            'similarity_matches' / f'run_{timestamp}')

def get_default_queries_file() -> Path:
    """Get default queries file path."""
    minhash_dir = Path(__file__).parent
    behaviors_file = minhash_dir / 'behaviors.yml'
    if not behaviors_file.exists():
        raise FileNotFoundError(f"Default behaviors file not found at {behaviors_file}")
    return behaviors_file

def main():
    parser = argparse.ArgumentParser(
        description="Process parquet files by partitioning files among processes. "
                    "Each process loads its group, checks passages against categorized behavior queries from YAML, "
                    "and outputs matches to a file with the same naming convention."
    )
    parser.add_argument("--input_dir", type=str, required=False,
                        help="Directory containing input parquet files. If not provided, will use default dataset.")
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="Directory to save output matched parquet files. Defaults to data/results/similarity_matches/run_TIMESTAMP")
    parser.add_argument("--min_file", type=str, default=None,
                        help="Minimum filename (without extension, lexicographically) to process.")
    parser.add_argument("--max_file", type=str, default=None,
                        help="Maximum filename (without extension, lexicographically) to process.")
    parser.add_argument("--passage_column", type=str, default="text",
                        help="Name of the column containing passages in the parquet files.")
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Number of permutations for MinHash.")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Minimum Jaccard similarity (approximate) to consider a passage matching a query.")
    parser.add_argument("--queries_yaml", type=str,
                        default=None,
                        help="YAML file containing behavior queries. Defaults to behaviors.yml in script directory.")
    parser.add_argument("--source_dataset", type=str,
                        default='Asap7772/open-web-math-processed-v2',
                        help='HuggingFace dataset to use as source if no input_dir provided')
    parser.add_argument("--max_dataset_gb", type=float, default=10.0,
                        help="Maximum dataset size in GB to process")
    args = parser.parse_args()

    # Setup output directory and stats tracker
    output_dir = Path(args.output_dir) if args.output_dir else get_default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    stats_tracker = StatsTracker(output_dir)

    # Create a multiprocessing Manager for shared checkpointing
    manager = Manager()
    shared_completed = manager.list(load_checkpoint(output_dir))
    checkpoint_lock = manager.Lock()

    if shared_completed:
        logger.info(f"Found checkpoint with {len(shared_completed)} completed files")

    # Resolve queries file path
    queries_file = Path(args.queries_yaml) if args.queries_yaml else get_default_queries_file()
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found at {queries_file}")

    # Save run configuration
    config = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': args.input_dir,
        'source_dataset': args.source_dataset,
        'num_perm': args.num_perm,
        'similarity_threshold': args.similarity_threshold,
        'passage_column': args.passage_column,
        'queries_file': str(queries_file),
    }
    with open(output_dir / 'run_config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup NLTK data
    setup_nltk()

    # Load categorized queries from the YAML file
    print("Loading categorized queries from YAML...")
    query_categories = load_queries_from_yaml(queries_file)
    total_queries = sum(len(queries) for queries in query_categories.values())
    print(f"Loaded {total_queries} queries across {len(query_categories)} categories from {queries_file}")

    # Save queries used for this run
    with open(output_dir / 'queries_used.yaml', 'w') as f:
        yaml.dump(query_categories, f)

    # Build the query index and signatures
    query_index, query_signatures, query_to_category = build_query_index(query_categories, args.num_perm)
    print("Query index built.")

    # Get input files - either from input_dir or by downloading dataset
    if args.input_dir:
        file_paths = list_input_files(args.input_dir, args.min_file, args.max_file)
    else:
        # Use default dataset location in project structure
        default_cache_dir = str(Path(__file__).parent.parent / 'data' / 'pretrained_data' / 'open-web-math')
        file_paths = ensure_dataset_cached(args.source_dataset, default_cache_dir, args.max_dataset_gb)

    # Filter out already completed files using shared_completed list
    file_paths = [f for f in file_paths if f not in shared_completed]

    if not file_paths:
        if shared_completed:
            logger.info("All files have been processed already!")
            return
        raise ValueError("No input files found matching the criteria.")

    print(f"Found {len(file_paths)} files to process")
    if shared_completed:
        print(f"Skipping {len(shared_completed)} already completed files")

    # Rest of processing remains the same
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
            str(output_dir),  # Convert Path to str for multiprocessing
            stats_tracker,
            shared_completed,
            checkpoint_lock
        ))

    # Launch the workers.
    start_time = time.time()
    with Pool(num_workers) as pool:
        pool.map(process_file_group, worker_args)

    runtime = time.time() - start_time
    print(f"Processing complete in {runtime:.2f} seconds.")

    # Save summary stats
    summary = {
        'runtime_seconds': runtime,
        'num_input_files': len(file_paths),
        'num_workers': num_workers,
    }
    with open(output_dir / 'run_summary.yaml', 'w') as f:
        yaml.dump(summary, f)

    logger.info(f"All results saved to: {output_dir}")

if __name__ == "__main__":
    main()
