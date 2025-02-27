"""
adapted from: https://github.com/eddycmu/demystify-long-cot/tree/e90cf9b30203bb74b1ea4ba008fa54bf731ad1a4

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
import pickle
import hashlib
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
from multiprocessing import Pool, cpu_count, Manager, Lock, Queue, Process
from datasketch import MinHash, MinHashLSHForest
import nltk
from nltk.tokenize import sent_tokenize
from loguru import logger
import pyarrow.parquet as pq
import pyarrow as pa
from functools import lru_cache

try:
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    logger.warning("Will attempt to continue but sentence tokenization may fail")


class Timer:
    """smol ctx mgr for timing code blks"""
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        self.logger.debug(f"⏱️ {self.name} took {self.duration:.4f} seconds")
        return False


def setup_nltk():
    """setup nltk data w/o drama"""
    try:
        nltk.data.find('punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            logger.warning("Will attempt to continue but sentence tokenization may fail")


def get_file_size_gb(file_path: str) -> float:
    """get file sz in gb"""
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)


def get_memory_usage_gb() -> float:
    """get current mem usage in gb, for monitoring resource usage"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)


def ensure_dataset_cached(dataset_name: str, cache_dir: str, max_size_gb: float = 10.0) -> list[str]:
    """
    cached locally or download
    - check if dataset exists locally
    - if not, dl w/ streaming to check sz
    - save in chunks to avoid mem issues
    - stop if exceeds max_size_gb
    returns: list of parquet file paths
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists() or not any(cache_path.glob("*.parquet")):
        logger.info(f"No cached dataset found at {cache_dir}, downloading {dataset_name}...")
        os.makedirs(cache_dir, exist_ok=True)

        # download with streaming to check size
        ds = datasets.load_dataset(dataset_name, split='train', streaming=True)

        # save in chunks of 100k examples
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

        # save any remaining examples and update size
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
        # check existing files
        parquet_files = list(cache_path.glob("*.parquet"))
        total_size_gb = sum(get_file_size_gb(str(f)) for f in parquet_files)
        if total_size_gb > max_size_gb:
            logger.warning(f"Cached dataset is {total_size_gb:.2f}GB, exceeding {max_size_gb}GB limit!")
        return [str(f) for f in parquet_files]


# --- helper funcs ---

def sanitize_filename(s):
    """sanitize str for file/dir name"""
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


# optimized for single-pass text processing
def clean_text(text):
    """
    clean txt by lowercasing & replacing non-word chars w/ spaces
    used for both prefiltering and shingle gen prep
    """
    return re.sub(r'\W+', ' ', text.lower())


# cached shingle generation - reduced complexity since text is pre-cleaned
@lru_cache(maxsize=10000)
def get_shingles(text, shingle_size):
    """
    create shingles from pre-cleaned txt
    assumes txt already lowercased & cleaned
    returns frozenset for hashability in cache
    """
    tokens = text.split()
    if len(tokens) < shingle_size:
        return frozenset(tokens)
    return frozenset(' '.join(tokens[i:i + shingle_size]) for i in range(len(tokens) - shingle_size + 1))


# fast sentence tokenization alternatives
def fast_sentence_tokenize(text, mode='balanced'):
    """
    faster alts to nltk's sent_tokenize

    args:
        text: txt to split into sentences
        mode: 'fast' for max speed, 'balanced' for better accuracy,
              'accurate' for nltk's tokenizer

    returns:
        list of sentences
    """
    if mode == 'accurate':
        return sent_tokenize(text)

    elif mode == 'balanced':
        # Replace common abbreviations with placeholders
        text = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)', lambda m: m.group().replace('.', '@'), text)

        # Use a simplified approach that doesn't rely on variable-width lookbehinds
        # First, add markers at potential sentence boundaries
        text = re.sub(r'([.!?])([\'")]*)(\s+)', r'\1\2<<SENT>>\3', text)

        # Handle common exceptions (don't split after these)
        text = re.sub(r'\b(Fig|etc|vs|i\.e|e\.g)\.<<SENT>>', r'\1.', text)

        # Split on the markers
        sentences = text.split('<<SENT>>')

        # Restore abbreviations and strip whitespace
        return [s.replace('@', '.').strip() for s in sentences if s.strip()]

    elif mode == 'fast':
        # Very simple split on punctuation + space
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'(?<=[.!?]) +', line)
            sentences.extend([p.strip() for p in parts if p.strip()])
        return sentences

    else:
        raise ValueError(f"Unknown tokenization mode: {mode}")


# extract common terms for fast filtering
def extract_common_terms(query_categories, min_freq=2):
    """
    extract common terms from queries for fast pre-filtering
    returns set of terms appearing at least min_freq times
    """
    all_terms = Counter()
    for category, queries in query_categories.items():
        for query in queries:
            terms = clean_text(query).split()
            all_terms.update(terms)

    # Keep terms that appear at least min_freq times
    return {term for term, count in all_terms.items() if count >= min_freq}


# cached minhash index management
def get_query_cache_path(queries_file, output_dir, num_perm, num_bands):
    """
    gen cache path for query idx based on params
    path includes hash of queries file content for versioning
    """
    # Create a hash of the queries file content for versioning
    with open(queries_file, 'rb') as f:
        queries_hash = hashlib.md5(f.read()).hexdigest()[:10]

    # Create a descriptive filename
    cache_filename = f"query_index_{queries_hash}_perm{num_perm}_bands{num_bands}.pkl"
    return os.path.join(output_dir, cache_filename)


def try_load_query_index(queries_file, output_dir, num_perm, num_bands):
    """
    try to load cached query idx if exists
    returns none if no valid cache exists
    """
    cache_path = get_query_cache_path(queries_file, output_dir, num_perm, num_bands)

    if not os.path.exists(cache_path):
        logger.info(f"No cached query index found at {cache_path}")
        return None

    try:
        logger.info(f"Loading cached query index from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)

        # Verify the cache has all required components
        required_keys = ['query_index', 'query_signatures', 'query_to_category',
                         'query_bands', 'band_to_queries']
        if not all(key in cached_data for key in required_keys):
            logger.warning("Cached query index is missing required components")
            return None

        logger.info("Successfully loaded cached query index")
        return (
            cached_data['query_index'],
            cached_data['query_signatures'],
            cached_data['query_to_category'],
            cached_data['query_bands'],
            cached_data['band_to_queries']
        )
    except Exception as e:
        logger.warning(f"Failed to load cached query index: {e}")
        return None


def save_query_index(queries_file, output_dir, num_perm, num_bands, query_index,
                     query_signatures, query_to_category, query_bands, band_to_queries):
    """
    save query idx to cache file for faster loading in future runs
    """
    cache_path = get_query_cache_path(queries_file, output_dir, num_perm, num_bands)

    try:
        logger.info(f"Saving query index to cache at {cache_path}")

        cached_data = {
            'query_index': query_index,
            'query_signatures': query_signatures,
            'query_to_category': query_to_category,
            'query_bands': query_bands,
            'band_to_queries': band_to_queries
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Query index successfully cached")
        return True
    except Exception as e:
        logger.warning(f"Failed to cache query index: {e}")
        return False


# optimized minhash generation with inverted band-to-queries index
def build_query_index(query_categories, num_perm, num_bands=10, shingle_size=3):
    """
    build optimized minhash idx w/ inverted band-to-queries mapping

    returns tuple of:
      - query_index: minhashLSHforest built from all queries
      - query_signatures: dict mapping query str to minhash obj
      - query_to_category: dict mapping query str to category name
      - query_bands: dict mapping query str to signature bands
      - band_to_queries: inverted idx mapping band hash to query list
    """
    with Timer("build_query_index", logger):
        query_signatures = {}
        query_to_category = {}
        query_index = MinHashLSHForest(num_perm=num_perm)
        query_bands = {}
        band_to_queries = defaultdict(list)  # New inverted index

        for category, queries in query_categories.items():
            for query in queries:
                # Clean the query text once
                cleaned_query = clean_text(query)

                m = MinHash(num_perm=num_perm)
                shingles = get_shingles(cleaned_query, shingle_size)
                for shingle in shingles:
                    m.update(shingle.encode("utf8"))

                # Store in the index
                query_signatures[query] = m
                query_to_category[query] = category
                query_index.add(query, m)

                # Pre-compute bands for faster initial matching
                bands = []
                signature = m.digest()
                band_size = len(signature) // num_bands
                for i in range(num_bands):
                    start = i * band_size
                    end = min(start + band_size, len(signature))
                    band_hash = hash(tuple(signature[start:end]))
                    bands.append(band_hash)

                    # Add to inverted index
                    band_to_queries[band_hash].append(query)

                query_bands[query] = bands

        query_index.index()

        # Convert defaultdict to regular dict for pickling
        band_to_queries_dict = dict(band_to_queries)

        logger.info(f"Built index with {len(query_signatures)} queries and {len(band_to_queries_dict)} unique bands")

        return query_index, query_signatures, query_to_category, query_bands, band_to_queries_dict


def partition_list(lst, num_partitions):
    """
    split list into n roughly equal parts
    """
    k, m = divmod(len(lst), num_partitions)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(num_partitions)]


def list_input_files(input_dir, min_file=None, max_file=None):
    """
    list all parquet files in input_dir between min_file and max_file
    (lexicographically). files sorted by filename (w/o extension)
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
    load categorized behavior queries from yaml
    expected format:
      verification_queries:
        - "query string one"
      backtracking_queries:
        - "another query string"
      ...etc
    returns: dict mapping category names to query lists
    """
    # normalize path extension if not provided
    if not os.path.exists(yaml_path):
        alt_ext = '.yml' if yaml_path.endswith('.yaml') else '.yaml'
        alt_path = yaml_path[:-len(os.path.splitext(yaml_path)[1])] + alt_ext
        if os.path.exists(alt_path):
            yaml_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find YAML file at {yaml_path} or with alternative extension {alt_ext}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # expected categories
    expected_categories = [
        "verification_queries",
        "backtracking_queries",
        "subgoal_queries",
        "backward_chaining_queries",
        "clarify_queries",
        "pivot_queries"
    ]

    # validate categories exist
    for category in expected_categories:
        if category not in data:
            raise ValueError(f"Missing required category '{category}' in YAML file")
        if not data[category]:
            raise ValueError(f"Category '{category}' is empty in YAML file")

    return data


# --- adaptive batch sizing ---

class AdaptiveBatchSizer:
    """
    dynamically adjusts batch size based on mem usage & processing time
    maintains efficient mem usage while maximizing throughput
    """
    def __init__(self, initial_batch_size=5000, target_memory_gb=3.0,
                 min_batch_size=1000, max_batch_size=20000):
        self.current_batch_size = initial_batch_size
        self.target_memory_gb = target_memory_gb
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.last_adjustment_time = time.time()
        self.adjustment_interval = 30  # seconds between adjustments

        # Performance tracking
        self.processing_times = []
        self.rows_per_second = []

    def get_batch_size(self):
        """get current recommended batch size"""
        return self.current_batch_size

    def update(self, rows_processed, processing_time):
        """
        update batch sizer with info about last batch

        args:
            rows_processed: num rows processed in last batch
            processing_time: time taken to process batch in seconds
        """
        # Track processing metrics
        self.processing_times.append(processing_time)
        rows_per_sec = rows_processed / processing_time if processing_time > 0 else 0
        self.rows_per_second.append(rows_per_sec)

        # Only adjust periodically
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return

        # Check current memory usage
        current_memory_gb = get_memory_usage_gb()

        # Adjust batch size based on memory usage and processing speed
        if current_memory_gb > self.target_memory_gb * 1.2:
            # Memory usage too high - decrease batch size
            adjustment_factor = 0.8
            logger.info(f"Memory usage high ({current_memory_gb:.2f}GB) - decreasing batch size")
        elif current_memory_gb < self.target_memory_gb * 0.7:
            # Memory usage low - increase batch size if processing speed is good
            if len(self.rows_per_second) > 1 and self.rows_per_second[-1] >= self.rows_per_second[-2] * 0.9:
                adjustment_factor = 1.2
                logger.info(f"Memory usage low ({current_memory_gb:.2f}GB) - increasing batch size")
            else:
                adjustment_factor = 1.0
        else:
            # Memory usage within target range - maintain batch size
            adjustment_factor = 1.0

        # Apply adjustment with limits
        new_batch_size = int(self.current_batch_size * adjustment_factor)
        new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))

        if new_batch_size != self.current_batch_size:
            logger.info(f"Adjusting batch size from {self.current_batch_size} to {new_batch_size}")
            self.current_batch_size = new_batch_size

        # Reset metrics for next interval
        self.last_adjustment_time = current_time
        if len(self.processing_times) > 10:
            self.processing_times = self.processing_times[-10:]
            self.rows_per_second = self.rows_per_second[-10:]

    def get_stats(self):
        """get current processing stats"""
        if not self.processing_times:
            return {"avg_rows_per_second": 0}

        avg_rows_per_sec = sum(self.rows_per_second) / len(self.rows_per_second)
        return {
            "current_batch_size": self.current_batch_size,
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times),
            "avg_rows_per_second": avg_rows_per_sec,
            "estimated_rows_per_hour": avg_rows_per_sec * 3600
        }


# --- memory-efficient pyarrow output ---

def write_results_with_pyarrow(matched_rows, output_file):
    """
    write matched rows directly to parquet using pyarrow for mem efficiency

    args:
        matched_rows: list of dicts containing matched data
        output_file: path to output parquet file
    """
    if not matched_rows:
        logger.warning(f"No matched rows to write to {output_file}")
        return

    try:
        # Convert to PyArrow table directly
        table = pa.Table.from_pylist(matched_rows)

        # Write using PyArrow with compression
        pq.write_table(
            table,
            output_file,
            compression='snappy',  # Faster than default
            use_dictionary=True,   # Better compression for repeated values
            version='2.6'          # Latest stable version
        )

        logger.info(f"Saved {len(matched_rows)} matches to {output_file} using PyArrow")

        # Help Python's GC by explicitly deleting variables
        del table

    except Exception as e:
        logger.error(f"Error writing results with PyArrow: {e}")

        # Fallback to pandas if PyArrow fails
        try:
            logger.warning("Falling back to pandas for writing results")
            df = pd.DataFrame(matched_rows)
            df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(matched_rows)} matches using pandas fallback")
        except Exception as e2:
            logger.error(f"Pandas fallback also failed: {e2}")
            raise


# --- checkpointing and stats tracking ---

def get_checkpoint_file(output_dir: Path) -> Path:
    """get path to checkpoint file"""
    return output_dir / "checkpoint.yaml"


def save_checkpoint(output_dir: Path, completed_files: list[str]):
    """save checkpoint of completed files"""
    checkpoint_file = get_checkpoint_file(output_dir)
    with open(checkpoint_file, 'w') as f:
        yaml.dump({'completed_files': completed_files}, f)


def load_checkpoint(output_dir: Path) -> list[str]:
    """load checkpoint of completed files"""
    checkpoint_file = get_checkpoint_file(output_dir)
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = yaml.safe_load(f)
            return checkpoint.get('completed_files', [])
    return []


class StatsTracker:
    """track and update stats for matching results"""
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
        self.last_update_time = time.time()
        self.update_interval = 10  # seconds between updates
        self.lock = Lock()
        self.load_existing_stats()

    def load_existing_stats(self):
        """load existing stats if they exist"""
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)

    def update(self, new_rows: int, matches_by_category: dict):
        """update stats with new batch of processed rows"""
        current_time = time.time()

        with self.lock:
            self.stats['total_rows_processed'] += new_rows

            # update category matches
            for category, count in matches_by_category.items():
                self.stats['matches_by_category'][category] += count

            # update total matches
            self.stats['total_rows_matched'] = sum(self.stats['matches_by_category'].values())

            # update percentages
            if self.stats['total_rows_processed'] > 0:
                total = self.stats['total_rows_processed']
                self.stats['category_percentages'] = {
                    cat: (count / total) * 100
                    for cat, count in self.stats['matches_by_category'].items()
                }
                self.stats['overall_match_percentage'] = (self.stats['total_rows_matched'] / total) * 100

            self.stats['last_update'] = datetime.now().isoformat()

            # Only save and display periodically to reduce overhead
            if current_time - self.last_update_time >= self.update_interval:
                self.save()
                self.display()
                self.last_update_time = current_time

    def save(self):
        """save current stats to file"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def display(self):
        """display current stats"""
        logger.info("\n=== Current Processing Statistics ===")
        logger.info(f"Total rows processed: {self.stats['total_rows_processed']:,}")
        logger.info(f"Total rows with matches: {self.stats['total_rows_matched']:,}")
        logger.info("\nMatches by category:")
        for cat, count in self.stats['matches_by_category'].items():
            percentage = self.stats['category_percentages'].get(cat, 0)
            logger.info(f"  {cat}: {count:,} ({percentage:.2f}%)")
        logger.info(f"\nOverall match rate: {self.stats.get('overall_match_percentage', 0):.2f}%")
        logger.info("=====================================")


# --- optimized batch processing ---

def process_batch(rows, passage_column, common_terms, query_index,
                 query_signatures, query_to_category, band_to_queries,
                 similarity_threshold, num_perm, tokenization_mode='balanced',
                 num_bands=10, enable_profiling=False, shingle_size=3):
    """
    process batch of rows efficiently using optimized approach

    returns tuple of (matched_rows, match_stats, timing_stats)
    """
    matched_rows = []
    matches_by_category = defaultdict(int)

    # Initialize timing stats for performance analysis
    timing_stats = defaultdict(float) if enable_profiling else None

    for row in rows:
        # Extract passage text
        if enable_profiling:
            with Timer("extract_passage", logger) as t:
                if isinstance(row, dict):
                    passage = row.get(passage_column, "")
                else:  # namedtuple from itertuples
                    passage = getattr(row, passage_column, "")
                if not passage:
                    continue
                timing_stats["extract_passage"] += t.duration
        else:
            if isinstance(row, dict):
                passage = row.get(passage_column, "")
            else:  # namedtuple from itertuples
                passage = getattr(row, passage_column, "")
            if not passage:
                continue

        # Clean text once for both pre-filtering and later processing
        if enable_profiling:
            with Timer("text_cleaning", logger) as t:
                cleaned_passage = clean_text(passage)
                timing_stats["text_cleaning"] += t.duration
        else:
            cleaned_passage = clean_text(passage)

        # Pre-filter using common terms
        if enable_profiling:
            with Timer("prefiltering", logger) as t:
                passage_terms = set(cleaned_passage.split())
                has_common_terms = bool(passage_terms.intersection(common_terms))
                timing_stats["prefiltering"] += t.duration
        else:
            passage_terms = set(cleaned_passage.split())
            has_common_terms = bool(passage_terms.intersection(common_terms))

        if not has_common_terms:
            continue  # Skip if no common terms

        # Tokenize into sentences using the selected mode
        if enable_profiling:
            with Timer("sentence_tokenization", logger) as t:
                sentences = fast_sentence_tokenize(passage, mode=tokenization_mode)
                timing_stats["sentence_tokenization"] += t.duration
        else:
            sentences = fast_sentence_tokenize(passage, mode=tokenization_mode)

        matched_by_category = {}

        for sentence in sentences:
            # Skip very short sentences - they're unlikely to match
            if len(sentence.split()) < 3:
                continue

            # Clean sentence once for processing
            if enable_profiling:
                with Timer("sentence_cleaning", logger) as t:
                    cleaned_sentence = clean_text(sentence)
                    timing_stats["sentence_cleaning"] += t.duration
            else:
                cleaned_sentence = clean_text(sentence)

            # Create MinHash signature for the sentence
            if enable_profiling:
                with Timer("minhash_creation", logger) as t:
                    m_sentence = MinHash(num_perm=num_perm)
                    shingles = get_shingles(cleaned_sentence, shingle_size)
                    for shingle in shingles:
                        m_sentence.update(shingle.encode("utf8"))
                    timing_stats["minhash_creation"] += t.duration
            else:
                m_sentence = MinHash(num_perm=num_perm)
                shingles = get_shingles(cleaned_sentence, shingle_size)
                for shingle in shingles:
                    m_sentence.update(shingle.encode("utf8"))

            # Skip further processing if not enough shingles
            if len(shingles) < 2:
                continue

            # Generate bands for the sentence MinHash
            if enable_profiling:
                with Timer("band_generation", logger) as t:
                    sentence_bands = []
                    signature = m_sentence.digest()
                    band_size = len(signature) // num_bands
                    for i in range(num_bands):
                        start = i * band_size
                        end = min(start + band_size, len(signature))
                        sentence_bands.append(hash(tuple(signature[start:end])))
                    timing_stats["band_generation"] += t.duration
            else:
                sentence_bands = []
                signature = m_sentence.digest()
                band_size = len(signature) // num_bands
                for i in range(num_bands):
                    start = i * band_size
                    end = min(start + band_size, len(signature))
                    sentence_bands.append(hash(tuple(signature[start:end])))

            # Use inverted index for fast candidate identification
            if enable_profiling:
                with Timer("candidate_identification", logger) as t:
                    candidate_queries = set()
                    for band in sentence_bands:
                        if band in band_to_queries:
                            candidate_queries.update(band_to_queries[band])
                    timing_stats["candidate_identification"] += t.duration
            else:
                candidate_queries = set()
                for band in sentence_bands:
                    if band in band_to_queries:
                        candidate_queries.update(band_to_queries[band])

            # If no candidates from band matching, try LSH Forest query as fallback
            if not candidate_queries and len(shingles) >= 3:
                if enable_profiling:
                    with Timer("lsh_forest_query", logger) as t:
                        candidates = query_index.query(m_sentence, 5)  # Top 5 matches
                        candidate_queries.update(candidates)
                        timing_stats["lsh_forest_query"] += t.duration
                else:
                    candidates = query_index.query(m_sentence, 5)  # Top 5 matches
                    candidate_queries.update(candidates)

            # Verify candidates with exact Jaccard similarity
            if enable_profiling:
                with Timer("similarity_verification", logger) as t:
                    for candidate in candidate_queries:
                        sim = m_sentence.jaccard(query_signatures[candidate])
                        if sim >= similarity_threshold:
                            category = query_to_category[candidate]
                            if category not in matched_by_category:
                                matched_by_category[category] = set()
                                matches_by_category[category] += 1
                            matched_by_category[category].add(candidate)
                    timing_stats["similarity_verification"] += t.duration
            else:
                for candidate in candidate_queries:
                    sim = m_sentence.jaccard(query_signatures[candidate])
                    if sim >= similarity_threshold:
                        category = query_to_category[candidate]
                        if category not in matched_by_category:
                            matched_by_category[category] = set()
                            matches_by_category[category] += 1
                        matched_by_category[category].add(candidate)

        # If we found matches, add to results
        if matched_by_category:
            if enable_profiling:
                with Timer("result_formatting", logger) as t:
                    # Convert row to dict based on its type
                    if hasattr(row, '_asdict'):  # namedtuple
                        row_dict = row._asdict()
                    elif hasattr(row, '_fields'):  # custom tuple-like
                        row_dict = {field: getattr(row, field) for field in row._fields}
                    else:  # assume it's already a dict-like structure
                        row_dict = dict(row)

                    # Add matching information
                    for category, matches in matched_by_category.items():
                        row_dict[f"matched_{category}"] = ", ".join(sorted(matches))

                    matched_rows.append(row_dict)
                    timing_stats["result_formatting"] += t.duration
            else:
                # Convert row to dict based on its type
                if hasattr(row, '_asdict'):  # namedtuple
                    row_dict = row._asdict()
                elif hasattr(row, '_fields'):  # custom tuple-like
                    row_dict = {field: getattr(row, field) for field in row._fields}
                else:  # assume it's already a dict-like structure
                    row_dict = dict(row)

                # Add matching information
                for category, matches in matched_by_category.items():
                    row_dict[f"matched_{category}"] = ", ".join(sorted(matches))

                matched_rows.append(row_dict)

    return matched_rows, dict(matches_by_category), timing_stats


# --- task queue for worker parallelism ---

def result_writer_process(result_queue, output_dir, stats_tracker, flush_threshold=1000):
    """
    dedicated process for writing results to disk
    separates i/o from processing to improve throughput
    """
    logger.info("Result writer process started")

    # Track accumulated results by file
    file_results = defaultdict(list)

    try:
        while True:
            try:
                # Get next result or None if queue is marked as done
                result = result_queue.get()
                if result is None:
                    # Special signal to finish
                    break

                batch_rows, output_file, batch_stats, rows_processed = result

                # Update stats
                stats_tracker.update(rows_processed, batch_stats)

                # Accumulate results
                file_results[output_file].extend(batch_rows)

                # Flush to disk when we have enough for a file
                for file_path, rows in list(file_results.items()):
                    if len(rows) >= flush_threshold:
                        write_results_with_pyarrow(rows, file_path)
                        file_results[file_path] = []

            except Exception as e:
                logger.error(f"Error in result writer: {e}")
                import traceback
                logger.error(traceback.format_exc())

    except KeyboardInterrupt:
        logger.info("Result writer received keyboard interrupt")
    finally:
        # Write any remaining results
        for file_path, rows in file_results.items():
            if rows:
                write_results_with_pyarrow(rows, file_path)

        logger.info("Result writer process finished")


def worker_process(task_queue, result_queue, worker_id, shared_args):
    """
    worker process that takes tasks from queue and processes them
    """
    # Unpack shared arguments
    (passage_column, num_perm, query_index, query_signatures,
     query_to_category, band_to_queries, common_terms, similarity_threshold,
     output_dir, batch_size, num_bands, tokenization_mode, enable_profiling,
     shingle_size) = shared_args

    logger.info(f"Worker {worker_id} started")

    try:
        while True:
            # Get next task or exit if queue is empty
            try:
                file_path, chunk_index = task_queue.get(block=False)
            except Exception:  # Queue.Empty or other error
                logger.info(f"Worker {worker_id} finished - no more tasks")
                break

            logger.info(f"Worker {worker_id} processing {file_path} chunk {chunk_index}")

            try:
                # Open the parquet file
                parquet_file = pq.ParquetFile(file_path)

                # Calculate chunk boundaries
                total_rows = parquet_file.metadata.num_rows
                chunks_total = (total_rows + batch_size - 1) // batch_size

                # Skip if chunk index is out of range
                if chunk_index >= chunks_total:
                    logger.warning(f"Chunk index {chunk_index} out of range for {file_path}")
                    continue

                # Read the specific chunk
                if chunk_index < parquet_file.num_row_groups:
                    table = parquet_file.read_row_group(chunk_index)
                    chunk_df = table.to_pandas(split_blocks=True, self_destruct=True)
                else:
                    # Handle case where chunk_index doesn't map directly to a row group
                    start_row = chunk_index * batch_size
                    end_row = min(start_row + batch_size, total_rows)
                    table = parquet_file.read(offset=start_row, length=end_row-start_row)
                    chunk_df = table.to_pandas(split_blocks=True, self_destruct=True)

                if passage_column not in chunk_df.columns:
                    logger.error(f"Column {passage_column} not found in {file_path}")
                    continue

                # Process this chunk using the optimized batch processor
                batch_rows, batch_stats, timing_stats = process_batch(
                    chunk_df.itertuples(index=False),
                    passage_column,
                    common_terms,
                    query_index,
                    query_signatures,
                    query_to_category,
                    band_to_queries,
                    similarity_threshold,
                    num_perm,
                    tokenization_mode,
                    num_bands,
                    enable_profiling,
                    shingle_size
                )

                # Log profiling information if enabled
                if enable_profiling and timing_stats:
                    logger.info(f"Worker {worker_id} timing stats for {file_path} chunk {chunk_index}:")
                    for operation, duration in sorted(timing_stats.items()):
                        logger.info(f"  {operation}: {duration:.4f}s")

                # If we found matches, queue them for writing
                if batch_rows:
                    base = os.path.splitext(os.path.basename(file_path))[0]
                    output_file = os.path.join(output_dir, f"{base}_chunk{chunk_index:05d}.parquet")

                    # Add result to result queue for writing
                    result_queue.put((batch_rows, output_file, batch_stats, chunk_df.shape[0]))
                else:
                    # No matches in this chunk, just update stats via result writer
                    result_queue.put(([], "", batch_stats, chunk_df.shape[0]))

                logger.info(f"Worker {worker_id} completed chunk {chunk_index} of {file_path}")

            except Exception as e:
                logger.error(f"Worker {worker_id} error processing {file_path} chunk {chunk_index}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Worker {worker_id} unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info(f"Worker {worker_id} exiting")


def get_default_output_dir() -> Path:
    """get default output dir with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(__file__).parent.parent / 'data' / 'results' /
            'similarity_matches' / f'run_{timestamp}')


def get_default_queries_file() -> Path:
    """get default queries file path"""
    minhash_dir = Path(__file__).parent
    behaviors_file = minhash_dir / 'behaviors.yml'
    if not behaviors_file.exists():
        raise FileNotFoundError(f"Default behaviors file not found at {behaviors_file}")
    return behaviors_file


def main():
    parser = argparse.ArgumentParser(
        description="optimized processor for matching passages against behavior queries"
    )
    parser.add_argument("--input_dir", type=str, required=False,
                        help="Directory containing input parquet files. If not provided, will use default dataset.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output matched parquet files. Defaults to data/results/similarity_matches/run_TIMESTAMP")
    parser.add_argument("--min_file", type=str, default=None,
                        help="Minimum filename (without extension, lexicographically) to process.")
    parser.add_argument("--max_file", type=str, default=None,
                        help="Maximum filename (without extension, lexicographically) to process.")
    parser.add_argument("--passage_column", type=str, default="text",
                        help="Name of the column containing passages in the parquet files.")
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Number of permutations for MinHash.")
    parser.add_argument("--similarity_threshold", type=float, default=0.2,
                        help="Minimum Jaccard similarity (approximate) to consider a passage matching a query.")
    parser.add_argument("--queries_yaml", type=str, default=None,
                        help="YAML file containing behavior queries. Defaults to behaviors.yml in script directory.")
    parser.add_argument("--source_dataset", type=str,
                        default='Asap7772/open-web-math-processed-v2',
                        help='HuggingFace dataset to use as source if no input_dir provided')
    parser.add_argument("--max_dataset_gb", type=float, default=10.0,
                        help="Maximum dataset size in GB to process")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Initial number of rows to process in each batch (may be adjusted dynamically)")
    parser.add_argument("--num_bands", type=int, default=10,
                        help="Number of bands for MinHash signature partitioning")
    parser.add_argument("--tokenization_mode", type=str, default='balanced',
                        choices=['accurate', 'balanced', 'fast'],
                        help="Sentence tokenization mode: accurate (NLTK), balanced (regex), fast (simple)")
    parser.add_argument("--flush_threshold", type=int, default=1000,
                        help="Number of matched rows to accumulate before writing to disk")
    parser.add_argument("--adaptive_batch_size", action="store_true",
                        help="Enable adaptive batch sizing based on memory usage")
    parser.add_argument("--target_memory_gb", type=float, default=8.0,
                        help="Target memory usage in GB for adaptive batch sizing")
    parser.add_argument("--enable_profiling", action="store_true",
                        help="Enable detailed performance profiling")
    parser.add_argument("--shingle_size", type=int, default=3,
                    help="Size of shingles for MinHash (default: 3)")
    args = parser.parse_args()

    # setup output dir and stats tracker
    output_dir = Path(args.output_dir) if args.output_dir else get_default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    stats_tracker = StatsTracker(output_dir)

    # resolve queries file path
    queries_file = Path(args.queries_yaml) if args.queries_yaml else get_default_queries_file()
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found at {queries_file}")

    # save run config
    config = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': args.input_dir,
        'source_dataset': args.source_dataset,
        'num_perm': args.num_perm,
        'similarity_threshold': args.similarity_threshold,
        'passage_column': args.passage_column,
        'queries_file': str(queries_file),
        'batch_size': args.batch_size,
        'num_bands': args.num_bands,
        'tokenization_mode': args.tokenization_mode,
        'adaptive_batch_size': args.adaptive_batch_size,
        'target_memory_gb': args.target_memory_gb,
        'flush_threshold': args.flush_threshold,
        'enable_profiling': args.enable_profiling,
        'shingle_size': args.shingle_size,
        'optimization_level': 'very high - using inverted idx, single-pass txt processing, adaptive sizing'
    }
    with open(output_dir / 'run_config.yaml', 'w') as f:
        yaml.dump(config, f)

    # setup nltk data
    setup_nltk()

    # load categorized queries from yaml
    logger.info("Loading categorized queries from YAML...")
    query_categories = load_queries_from_yaml(queries_file)
    total_queries = sum(len(queries) for queries in query_categories.values())
    logger.info(f"Loaded {total_queries} queries across {len(query_categories)} categories from {queries_file}")

    # save queries used for this run
    with open(output_dir / 'queries_used.yaml', 'w') as f:
        yaml.dump(query_categories, f)

    # extract common terms for fast pre-filtering
    common_terms = extract_common_terms(query_categories)
    logger.info(f"Extracted {len(common_terms)} common query terms for fast pre-filtering")

    # try to load cached query idx or build new one
    cached_index = try_load_query_index(queries_file, output_dir, args.num_perm, args.num_bands)

    if cached_index:
        logger.info("Using cached query index")
        query_index, query_signatures, query_to_category, query_bands, band_to_queries = cached_index
    else:
        # build optimized query idx w/ inverted band-to-query mapping
        logger.info("Building query index with optimized inverted indexing...")
        query_index, query_signatures, query_to_category, query_bands, band_to_queries = build_query_index(
            query_categories, args.num_perm, args.num_bands, args.shingle_size
        )

        # save for future use
        save_query_index(queries_file, output_dir, args.num_perm, args.num_bands,
                        query_index, query_signatures, query_to_category,
                        query_bands, band_to_queries)

    # get input files - either from input_dir or by downloading dataset
    if args.input_dir:
        file_paths = list_input_files(args.input_dir, args.min_file, args.max_file)
    else:
        # default dataset location
        default_cache_dir = str(Path(__file__).parent.parent / 'data' / 'pretrained_data' / 'open-web-math')
        file_paths = ensure_dataset_cached(args.source_dataset, default_cache_dir, args.max_dataset_gb)

    if not file_paths:
        raise ValueError("No input files found matching the criteria.")

    logger.info(f"Found {len(file_paths)} files to process")

    # create task queue w/ chunk-level granularity
    task_queue = Queue()
    result_queue = Queue()

    # populate task queue w/ file+chunk pairs
    logger.info("Preparing processing tasks...")
    for file_path in file_paths:
        try:
            # get num chunks in this file
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            batch_size = args.batch_size
            num_chunks = (total_rows + batch_size - 1) // batch_size

            logger.info(f"File {file_path} has {total_rows} rows, will be processed in {num_chunks} chunks")

            # add each chunk as separate task
            for chunk_idx in range(num_chunks):
                task_queue.put((file_path, chunk_idx))

        except Exception as e:
            logger.error(f"Error preparing tasks for {file_path}: {e}")

    logger.info(f"Created {task_queue.qsize()} processing tasks")

    # determine num workers
    num_workers = min(cpu_count(), task_queue.qsize())
    if num_workers < 1:
        num_workers = 1
    logger.info(f"Starting {num_workers} worker processes")

    # prep shared args for workers
    shared_args = (
        args.passage_column, args.num_perm, query_index,
        query_signatures, query_to_category, band_to_queries,
        common_terms, args.similarity_threshold, str(output_dir),
        args.batch_size, args.num_bands, args.tokenization_mode,
        args.enable_profiling, args.shingle_size
    )

    # start result writer process
    result_writer = Process(
        target=result_writer_process,
        args=(result_queue, str(output_dir), stats_tracker, args.flush_threshold)
    )
    result_writer.start()

    # start worker processes
    workers = []
    for worker_id in range(num_workers):
        p = Process(
            target=worker_process,
            args=(task_queue, result_queue, worker_id, shared_args)
        )
        workers.append(p)
        p.start()

    # wait for all workers to finish
    start_time = time.time()
    logger.info("All processes started, waiting for completion...")

    try:
        for worker in workers:
            worker.join()

        # signal result writer to finish
        result_queue.put(None)
        result_writer.join()

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received, terminating processes...")
        for worker in workers:
            worker.terminate()
        result_writer.terminate()
        raise

    runtime = time.time() - start_time
    logger.info(f"Processing complete in {runtime:.2f} seconds")

    # save summary stats
    summary = {
        'runtime_seconds': runtime,
        'num_input_files': len(file_paths),
        'num_workers': num_workers,
        'processed_chunks': task_queue.qsize()
    }
    with open(output_dir / 'run_summary.yaml', 'w') as f:
        yaml.dump(summary, f)

    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()