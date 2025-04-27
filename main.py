import time

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from boyer_moore import bm_search
from download_IMDB import create_cross_matrix_inputs, load_reviews
from horspool import horspool_search

# --- Settings ---

text_size_ranges = {'small': (900, 1100), 'medium': (9500, 10500), 'large': (99500, 100500)}

pattern_size_words = {'small': (1, 1), 'medium': (3, 4), 'large': (5, 7)}

num_iterations = 1000


# --- Functions ---

def is_correct_match(text, pattern, pos):
    return (pos != -1) and (text[pos:pos + len(pattern)] == pattern)


# --- Load IMDB Dataset ---

reviews, vocab = load_reviews()

# --- Main Experiment ---

all_results = []

for _ in tqdm(range(num_iterations), desc="Running IMDB Experiments"):
    input_matrix = create_cross_matrix_inputs(reviews, vocab, text_size_ranges, pattern_size_words)

    for entry in input_matrix:
        text = entry['text']
        pattern = entry['pattern']
        pattern_size = entry['pattern_size']
        text_size = entry['text_size']

        # Horspool Timing
        start = time.perf_counter_ns()
        horspool_pos = horspool_search(text, pattern)
        horspool_time_ns = time.perf_counter_ns() - start

        # Boyer-Moore Timing
        start = time.perf_counter_ns()
        bm_pos = bm_search(text, pattern)
        bm_time_ns = time.perf_counter_ns() - start

        # Correctness Checking
        horspool_correct = is_correct_match(text, pattern, horspool_pos)
        bm_correct = is_correct_match(text, pattern, bm_pos)

        # Store results
        all_results.append({'Pattern Size': pattern_size, 'Text Size': text_size, 'Horspool Found': horspool_correct,
            'Boyer-Moore Found': bm_correct, 'Horspool Time (ns)': horspool_time_ns, 'BM Time (ns)': bm_time_ns})

# --- Results Processing ---

df_results = pd.DataFrame(all_results)
df_results['Test Case'] = df_results['Pattern Size'] + ' pattern / ' + df_results['Text Size'] + ' text'

# Grouped averages
df_avg = df_results.groupby('Test Case').mean(numeric_only=True).reset_index()

print("\n=== Average IMDB Results ===\n")
print(df_avg)

# Show Accuracy of algs
total_tests = len(df_results)
horspool_correct_total = df_results['Horspool Found'].sum()
bm_correct_total = df_results['Boyer-Moore Found'].sum()

horspool_accuracy = (horspool_correct_total / total_tests) * 100
bm_accuracy = (bm_correct_total / total_tests) * 100

print("\n=== ACCURACY ===\n")
print(f"Horspool Accuracy: {horspool_accuracy:.2f}%")
print(f"Boyer-Moore Accuracy: {bm_accuracy:.2f}%")

plt.figure(figsize=(14, 8))

bar_width = 0.35
r1 = range(len(df_avg))
r2 = [x + bar_width for x in r1]

# Convert ns to μs for readability
plt.bar(r1, df_avg['Horspool Time (ns)'] / 1000, width=bar_width, label='Horspool (μs)')
plt.bar(r2, df_avg['BM Time (ns)'] / 1000, width=bar_width, label='Boyer-Moore (μs)')

plt.xlabel('Test Cases', fontsize=14)
plt.ylabel('Average Execution Time (microseconds)', fontsize=14)
plt.title(f'Horspool vs Boyer-Moore Timing Comparison ({num_iterations} IMDB runs)', fontsize=16)
plt.xticks([r + bar_width / 2 for r in range(len(df_avg))], df_avg['Test Case'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()

plt.savefig('datasets/imdb_experiment_plot.png')
print("\nPlot saved to 'datasets/imdb_experiment_plot.png'")

plt.show()

# Save results
# df_results.to_csv('datasets/imdb_experiment_all.csv', index=False)
# df_avg.to_csv('datasets/imdb_experiment_avg.csv', index=False)
# print("\nResults saved to 'datasets/imdb_experiment_all.csv' and 'datasets/imdb_experiment_avg.csv'")
