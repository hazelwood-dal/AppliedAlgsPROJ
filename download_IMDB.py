import os
import random
import tarfile

import pandas as pd
import requests


def download_imdb_dataset(dest_folder='datasets'):
    os.makedirs(dest_folder, exist_ok=True)
    imdb_csv_path = os.path.join(dest_folder, 'IMDB_Dataset.csv')
    vocab_path = os.path.join(dest_folder, 'aclImdb/imdb.vocab')

    if not os.path.exists(imdb_csv_path) or not os.path.exists(vocab_path):
        print("IMDB dataset or vocab not found. Downloading...")
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            print("Download complete. Extracting...")
            tar_file_path = os.path.join(dest_folder, 'aclImdb_v1.tar.gz')
            with open(tar_file_path, 'wb') as f:
                f.write(response.raw.read())

            with tarfile.open(tar_file_path, 'r:gz') as tar:
                tar.extractall(path=dest_folder)

            print("Extraction complete. Preparing CSV...")
            imdb_reviews_to_csv(dest_folder)
        else:
            raise Exception(f"Failed to download dataset! Status code: {response.status_code}")
    else:
        print("IMDB dataset and vocab found locally.")

    return imdb_csv_path, vocab_path


def imdb_reviews_to_csv(dest_folder):
    data = {'review': [], 'sentiment': []}
    for label in ['pos', 'neg']:
        for split in ['train', 'test']:
            dir_path = os.path.join(dest_folder, 'aclImdb', split, label)
            for filename in os.listdir(dir_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(dir_path, filename), encoding='utf8') as f:
                        data['review'].append(f.read())
                        data['sentiment'].append(label)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(dest_folder, 'IMDB_Dataset.csv'), index=False)
    print(f"CSV saved at {os.path.join(dest_folder, 'IMDB_Dataset.csv')}.")


def load_imdb_data(dest_folder='datasets', num_samples=10):
    imdb_csv_path = download_imdb_dataset(dest_folder)
    df = pd.read_csv(imdb_csv_path)
    reviews = df['review'].dropna().tolist()
    return random.sample(reviews, num_samples)


def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf8') as f:
        vocab = [line.strip() for line in f.readlines()]
    return vocab


def load_reviews(dest_folder='datasets'):
    imdb_csv_path, vocab_path = download_imdb_dataset(dest_folder)
    df = pd.read_csv(imdb_csv_path)

    reviews = df['review'].dropna().tolist()
    vocab = load_vocab(vocab_path)

    return reviews, vocab


# If not enough long reviews, build one
def build_large_text(reviews, target_length):
    text = ""
    while len(text) < target_length:
        text += random.choice(reviews) + " "
    return text[:target_length]


def create_cross_matrix_inputs(reviews, vocab, text_size_ranges, pattern_size_words):
    inputs = []

    # Prepare texts of different sizes
    texts = {'small': [], 'medium': [], 'large': []}

    for size, (low, high) in text_size_ranges.items():
        target_length = random.randint(low, high)
        suitable_reviews = [r for r in reviews if len(r) >= target_length]

        if suitable_reviews:
            chosen = random.choice(suitable_reviews)
            trimmed = chosen[:target_length]
        else:
            # Build Long review if needed
            trimmed = build_large_text(reviews, target_length)

        texts[size] = trimmed

    # Create cross-matrix combinations
    for pattern_size in ['small', 'medium', 'large']:
        for text_size in ['small', 'medium', 'large']:
            text = texts[text_size]

            # Generate pattern
            word_count = random.randint(*pattern_size_words[pattern_size])
            pattern = ' '.join(random.choices(vocab, k=word_count))

            # Insert pattern randomly into text
            insert_pos = random.randint(0, len(text))
            modified_text = f'{text[:insert_pos]} {pattern} {text[insert_pos:]}'

            inputs.append(
                {'pattern_size': pattern_size, 'text_size': text_size, 'text': modified_text, 'pattern': pattern})

    return inputs


if __name__ == "__main__":
    reviews, vocab = load_reviews()
    input_matrix = create_cross_matrix_inputs(reviews, vocab)

    for entry in input_matrix:
        print(f"\n--- {entry['pattern_size'].capitalize()} Pattern with {entry['text_size'].capitalize()} Text ---")
        print(f"Pattern: '{entry['pattern']}'")
        print(f"Text Length: {len(entry['text'])}")
        print(f"Text Preview: {entry['text'][:200]}...\n")
