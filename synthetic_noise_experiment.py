import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from boyer_moore import bm_search
from horspool import horspool_search

text_size_ranges = {'small': (900, 1100), 'medium': (9500, 10500), 'large': (99500, 100500)}

pattern_size_ranges = {'small': (3, 5), 'medium': (10, 15), 'large': (30, 40)}

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_iterations = 1000
combinations = [(p_size, t_size) for p_size in ['small', 'medium', 'large'] for t_size in ['small', 'medium', 'large']]


def generate_slightly_noisy_text(length, noise_ratio):
    text = []
    for _ in range(length):
        if random.random() < noise_ratio:
            text.append(random.choice('BCDEFGHIJKLMNOPQRSTUVWXYZ'))
        else:
            text.append('A')
    return ''.join(text)


def is_correct_match(text, pattern, pos):
    return (pos != -1) and (text[pos:pos + len(pattern)] == pattern)


all_results = []

for noise_ratio in noise_levels:
    print(f"\nRunning experiments with noise level {noise_ratio:.1f}")

    for _ in tqdm(range(num_iterations), desc=f"Noise {noise_ratio:.1f}"):
        for pattern_size_label, text_size_label in combinations:
            # Generate params for text gen
            text_low, text_high = text_size_ranges[text_size_label]
            text_length = random.randint(text_low, text_high)

            # Generate pattern
            pat_low, pat_high = pattern_size_ranges[pattern_size_label]
            pattern_length = random.randint(pat_low, pat_high)
            pattern = 'A' * pattern_length

            # Generate Text
            text = generate_slightly_noisy_text(text_length, noise_ratio)
            # Insert Pattern Randomly
            insert_pos = random.randint(0, len(text))
            text = text[:insert_pos] + pattern + text[insert_pos:]

            # Horspool Timing
            start = time.perf_counter_ns()
            horspool_pos = horspool_search(text, pattern)
            horspool_time_ns = time.perf_counter_ns() - start

            # Boyer-Moore Timing
            start = time.perf_counter_ns()
            bm_pos = bm_search(text, pattern)
            bm_time_ns = time.perf_counter_ns() - start

            # Check for correctly found string
            horspool_correct = is_correct_match(text, pattern, horspool_pos)
            bm_correct = is_correct_match(text, pattern, bm_pos)

            # Store results
            all_results.append(
                {'Noise Level': noise_ratio, 'Pattern Size': pattern_size_label, 'Text Size': text_size_label,
                    'Horspool Found': horspool_correct, 'Boyer-Moore Found': bm_correct,
                    'Horspool Time (ns)': horspool_time_ns, 'BM Time (ns)': bm_time_ns})

df_results = pd.DataFrame(all_results)
df_results['Test Case'] = df_results['Pattern Size'] + ' pattern / ' + df_results['Text Size'] + ' text'

# Grouped averages
df_avg = df_results.groupby(['Noise Level', 'Test Case']).mean(numeric_only=True).reset_index()

print("\n=== Average Noise-Level Results ===\n")
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

# Plot each combination
for test_case in df_avg['Test Case'].unique():
    df_sub = df_avg[df_avg['Test Case'] == test_case]

    plt.figure(figsize=(12, 6))
    plt.plot(df_sub['Noise Level'], df_sub['Horspool Time (ns)'] / 1000, marker='o', label='Horspool (μs)')
    plt.plot(df_sub['Noise Level'], df_sub['BM Time (ns)'] / 1000, marker='x', label='Boyer-Moore (μs)')

    plt.xlabel('Noise Level (%)', fontsize=14)
    plt.ylabel('Average Execution Time (microseconds)', fontsize=14)
    plt.title(f'Timing vs Noise Level - {test_case}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'datasets/noise_experiment_{test_case.replace("/", "_").replace(" ", "_")}.png')
    plt.close()

print("\nPlots saved for each Test Case.")

# df_results.to_csv('datasets/noise_experiment_all.csv', index=False)
# df_avg.to_csv('datasets/noise_experiment_avg.csv', index=False)
# print("\nResults saved to 'datasets/noise_experiment_all.csv' and 'datasets/noise_experiment_avg.csv'")

# Prepare pivot tables for heatmaps
pivot_horspool = df_avg.pivot(index='Test Case', columns='Noise Level', values='Horspool Time (ns)')
pivot_boyer_moore = df_avg.pivot(index='Test Case', columns='Noise Level', values='BM Time (ns)')

# Convert ns -> μs for better reading
pivot_horspool = pivot_horspool / 1000
pivot_boyer_moore = pivot_boyer_moore / 1000

# Plot Horspool Heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_horspool, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Time (μs)'})
plt.title("Horspool Timing vs Noise Level (Heatmap)", fontsize=18)
plt.xlabel("Noise Level", fontsize=14)
plt.ylabel("Pattern Size / Text Size", fontsize=14)
plt.tight_layout()
plt.savefig('datasets/noise_heatmap_horspool.png')
plt.show()

# Plot Boyer-Moore Heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_boyer_moore, annot=True, fmt=".1f", cmap="Oranges", cbar_kws={'label': 'Time (μs)'})
plt.title("Boyer-Moore Timing vs Noise Level (Heatmap)", fontsize=18)
plt.xlabel("Noise Level", fontsize=14)
plt.ylabel("Pattern Size / Text Size", fontsize=14)
plt.tight_layout()
plt.savefig('datasets/noise_heatmap_boyer_moore.png')
plt.show()

print("\nHeatmaps saved to 'datasets/noise_heatmap_horspool.png' and 'datasets/noise_heatmap_boyer_moore.png'")
