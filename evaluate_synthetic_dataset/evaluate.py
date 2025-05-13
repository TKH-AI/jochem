import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot configurations to use LaTeX font
# Set plot configurations to use LaTeX font and grey color scheme
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 14,      # Axis labels
    "axes.titlesize": 18,      # Title size
    "xtick.labelsize": 12,     # X-axis tick labels
    "ytick.labelsize": 12,     # Y-axis tick labels
    "legend.fontsize": 14,     # Legend font size
    "axes.facecolor": "#f0f0f0",
    "axes.edgecolor": "#b0b0b0",
    "xtick.color": "#707070",
    "ytick.color": "#707070",
    "text.color": "#303030"
})

# -------------------------
# Configuration Parameters
# -------------------------

# Paths to the real and synthetic datasets
REAL_DATA_PATHS = [
    "/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_LinkedIn_aligned.csv",
    "/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_Press_Release_aligned.csv",
    "/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_twitter_aligned.csv"
]

SYNTHETIC_DATA_PATHS = [
    "/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_linkedin_clean_llama.csv",
    "/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_press_release_clean_llama.csv",
    "/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_x_clean_llama.csv"
]

# Platforms to evaluate
PLATFORMS = ['X', 'LinkedIn', 'Press_Release']

# N-gram size for Vocabulary Statistics
NGRAM = 2  # For bigrams

# Sentence Transformer model for Cosine Similarity
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Output Excel report path
REPORT_PATH = "synthetic_data_evaluation_report.xlsx"

# -------------------------
# Helper Functions
# -------------------------

def load_datasets(real_paths, synthetic_paths):
    """
    Load and concatenate real and synthetic datasets from provided file paths.
    """
    real_dfs = []
    for path in real_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            real_dfs.append(df)
        else:
            print(f"Real data file not found: {path}")
    
    synthetic_dfs = []
    for path in synthetic_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            synthetic_dfs.append(df)
        else:
            print(f"Synthetic data file not found: {path}")
    
    # Concatenate all real and synthetic data
    real_data = pd.concat(real_dfs, ignore_index=True) if real_dfs else pd.DataFrame()
    synthetic_data = pd.concat(synthetic_dfs, ignore_index=True) if synthetic_dfs else pd.DataFrame()
    
    return real_data, synthetic_data

def identify_synthetic(real_data, synthetic_data, platforms):
    """
    Identify synthetic data by comparing real and synthetic datasets.
    Synthetic data is where real data has NaN in platform columns but synthetic data has non-NaN.
    """
    synthetic_flags = {}
    for platform in platforms:
        if platform not in real_data.columns or platform not in synthetic_data.columns:
            print(f"Platform column '{platform}' not found in datasets.")
            synthetic_flags[platform] = pd.Series([False]*len(synthetic_data))
            continue
        
        synthetic_flags[platform] = synthetic_data[platform].notna() & real_data[platform].isna()
    
    return synthetic_flags

def compute_vocabulary_statistics(synthetic_data, synthetic_flags, platforms, ngram=2):
    """
    Compute vocabulary size and n-gram frequency distribution for synthetic data.
    """
    results = {}
    for platform in platforms:
        synthetic_texts = synthetic_data.loc[synthetic_flags[platform], platform].dropna().tolist()
        
        if not synthetic_texts:
            results[platform] = {'vocab_size': None, f'{ngram}-gram_freq': None}
            continue
        
        vectorizer = CountVectorizer(ngram_range=(ngram, ngram), token_pattern=r'\b\w+\b')
        X = vectorizer.fit_transform(synthetic_texts)
        vocab_size = len(vectorizer.get_feature_names_out())
        ngram_freq = X.sum(axis=0).A1  # Sum over all documents
        freq_dict = dict(zip(vectorizer.get_feature_names_out(), ngram_freq))
        
        results[platform] = {
            'vocab_size': vocab_size,
            f'{ngram}-gram_freq': freq_dict
        }
    return results

def compute_cosine_similarity_within_synthetic(synthetic_data, synthetic_flags, platforms, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Compute average cosine similarity between synthetic texts to assess contextual diversity.
    Lower similarity indicates higher diversity.
    """
    model = SentenceTransformer(model_name)
    results = {}
    for platform in platforms:
        synthetic_texts = synthetic_data.loc[synthetic_flags[platform], platform].dropna().tolist()
        
        if len(synthetic_texts) < 2:
            # Cannot compute similarity with less than 2 samples
            results[platform] = None
            continue
        
        embeddings = model.encode(synthetic_texts, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
        
        # Compute pairwise cosine similarity
        cosine_sim_matrix = cosine_similarity(embeddings)
        
        # Exclude self-similarity by setting diagonal to 0
        np.fill_diagonal(cosine_sim_matrix, 0)
        
        # Compute average cosine similarity
        avg_cosine_sim = cosine_sim_matrix.sum() / (len(synthetic_texts) * (len(synthetic_texts) - 1))
        results[platform] = avg_cosine_sim
    return results

def compute_cosine_similarity_between_real_synthetic(real_data, synthetic_data, platforms, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Compute average cosine similarity between real and synthetic texts to assess similarity.
    Higher similarity indicates more similarity between datasets.
    """
    model = SentenceTransformer(model_name)
    results = {}
    for platform in platforms:
        real_texts = real_data[platform].dropna().tolist()
        synthetic_texts = synthetic_data[platform].dropna().tolist()
        
        if not real_texts or not synthetic_texts:
            results[platform] = None
            continue
        
        # Encode texts
        real_embeddings = model.encode(real_texts, convert_to_tensor=True, show_progress_bar=True)
        synthetic_embeddings = model.encode(synthetic_texts, convert_to_tensor=True, show_progress_bar=True)
        
        real_embeddings = real_embeddings.cpu().numpy()
        synthetic_embeddings = synthetic_embeddings.cpu().numpy()
        
        # Compute cosine similarity between all pairs
        cosine_sim_matrix = cosine_similarity(real_embeddings, synthetic_embeddings)
        
        # Compute average cosine similarity
        avg_cosine_sim = cosine_sim_matrix.mean()
        results[platform] = avg_cosine_sim
    return results

def compute_sample_distance_between_real_synthetic(real_data, synthetic_data, platforms, distance_metric='euclidean'):
    """
    Compute average sample distance between real and synthetic texts to assess dissimilarity.
    Higher distance indicates more dissimilarity between datasets.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    results = {}
    for platform in platforms:
        real_texts = real_data[platform].dropna().tolist()
        synthetic_texts = synthetic_data[platform].dropna().tolist()
        
        if not real_texts or not synthetic_texts:
            results[platform] = None
            continue
        
        # Encode texts
        real_embeddings = model.encode(real_texts, convert_to_tensor=True, show_progress_bar=True)
        synthetic_embeddings = model.encode(synthetic_texts, convert_to_tensor=True, show_progress_bar=True)
        
        real_embeddings = real_embeddings.cpu().numpy()
        synthetic_embeddings = synthetic_embeddings.cpu().numpy()
        
        # To make it computationally feasible, sample a subset if datasets are large
        max_samples = 1000  # Adjust based on available memory
        if real_embeddings.shape[0] > max_samples:
            real_embeddings = real_embeddings[np.random.choice(real_embeddings.shape[0], max_samples, replace=False)]
        if synthetic_embeddings.shape[0] > max_samples:
            synthetic_embeddings = synthetic_embeddings[np.random.choice(synthetic_embeddings.shape[0], max_samples, replace=False)]
        
        # Compute pairwise distances
        if distance_metric == 'euclidean':
            distance_matrix = np.linalg.norm(real_embeddings[:, np.newaxis] - synthetic_embeddings, axis=2)
        elif distance_metric == 'cosine':
            # Cosine distance is 1 - cosine similarity
            cosine_sim_matrix = cosine_similarity(real_embeddings, synthetic_embeddings)
            distance_matrix = 1 - cosine_sim_matrix
        else:
            raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'cosine'.")
        
        # Compute average distance
        avg_distance = distance_matrix.mean()
        results[platform] = avg_distance
    return results

def save_evaluation_report(vocab_stats, cosine_sim_within, real_synth_cosine_sim, sample_distance_results, platforms, report_path):
    """
    Save the evaluation results to an Excel file with separate sheets.
    """
    with pd.ExcelWriter(report_path) as writer:
        # Vocabulary Size
        vocab_size_df = pd.DataFrame({
            'Platform': platforms,
            'Vocabulary Size': [vocab_stats[platform]['vocab_size'] if vocab_stats[platform]['vocab_size'] else 'N/A' for platform in platforms]
        })
        vocab_size_df.to_excel(writer, sheet_name='Vocabulary_Size', index=False)
        
        # Top 5 Bigrams
        for platform in platforms:
            if vocab_stats[platform][f'{NGRAM}-gram_freq']:
                top_5 = sorted(vocab_stats[platform][f'{NGRAM}-gram_freq'].items(), key=lambda x: x[1], reverse=True)[:5]
                top_5_df = pd.DataFrame(top_5, columns=['Bigram', 'Frequency'])
                top_5_df.to_excel(writer, sheet_name=f'Top5_Bigrams_{platform}', index=False)
        
        # Cosine Similarity Within Synthetic Data
        cosine_sim_within_df = pd.DataFrame({
            'Platform': platforms,
            'Average Cosine Similarity Within Synthetic Data': [f"{cosine_sim_within[platform]:.4f}" if cosine_sim_within[platform] else 'N/A' for platform in platforms]
        })
        cosine_sim_within_df.to_excel(writer, sheet_name='Cosine_Similarity_Within_Synthetic', index=False)
        
        # Cosine Similarity Between Real and Synthetic Data
        cosine_sim_between_df = pd.DataFrame({
            'Platform': platforms,
            'Average Cosine Similarity Between Real and Synthetic Data': [f"{real_synth_cosine_sim[platform]:.4f}" if real_synth_cosine_sim[platform] else 'N/A' for platform in platforms]
        })
        cosine_sim_between_df.to_excel(writer, sheet_name='Cosine_Similarity_Between', index=False)
        
        # Sample Distance Between Real and Synthetic Data
        sample_distance_df = pd.DataFrame({
            'Platform': platforms,
            'Average Sample Distance (Euclidean)': [f"{sample_distance_results[platform]:.4f}" if sample_distance_results[platform] else 'N/A' for platform in platforms]
        })
        sample_distance_df.to_excel(writer, sheet_name='Sample_Distance_Between', index=False)
    
    print(f"Evaluation report saved to '{report_path}'.")

def visualize_vocabulary_statistics(vocab_stats, platforms, ngram=2):
    """
    Visualize vocabulary size and top n-grams for each platform.
    """
    fig, axs = plt.subplots(2, len(platforms), figsize=(20, 12))
    fig.suptitle('Vocabulary Statistics for Synthetic Data', fontsize=22)

    # Vocabulary Size
    vocab_sizes = [vocab_stats[platform]['vocab_size'] if vocab_stats[platform]['vocab_size'] else 0 for platform in platforms]
    sns.barplot(x=platforms, y=vocab_sizes, palette="viridis", ax=axs[0, 0])
    axs[0, 0].set_title('Vocabulary Size for Synthetic Data')
    axs[0, 0].set_ylabel('Number of Unique Bigrams')
    axs[0, 0].set_xlabel('Platform')

    # Top 5 N-grams per Platform
    for idx, platform in enumerate(platforms):
        if vocab_stats[platform][f'{ngram}-gram_freq']:
            top_5_ngrams = sorted(vocab_stats[platform][f'{ngram}-gram_freq'].items(), key=lambda x: x[1], reverse=True)[:5]
            ngrams, freqs = zip(*top_5_ngrams)
            sns.barplot(x=list(ngrams), y=list(freqs), palette="magma", ax=axs[1, idx])
            axs[1, idx].set_title(f'Top 5 Bigrams for {platform} Platform')
            axs[1, idx].set_ylabel('Frequency')
            axs[1, idx].set_xlabel('Bigrams')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust top margin to make room for the title
    plt.savefig("vocabulary_statistics.png")
    plt.show()

def visualize_cosine_similarity_and_sample_distance(cosine_sim_within, real_synth_cosine_sim, sample_distance_within, sample_distance_between, platforms):
    """
    Visualize cosine similarity and sample distance for each platform.
    """
    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Cosine Similarity and Sample Distance for Synthetic and Real Data', fontsize=22)

    # Set color palette to greyish
    palette = sns.color_palette("gray")

    # Cosine Similarity Within Synthetic Data
    cosine_sims_within = [cosine_sim_within[platform] if cosine_sim_within[platform] is not None else 0 for platform in platforms]
    sns.barplot(x=platforms, y=cosine_sims_within, palette=palette, ax=axs[0, 0])
    axs[0, 0].set_title('Average Cosine Similarity Within Synthetic Data')
    axs[0, 0].set_ylabel('Average Cosine Similarity')
    axs[0, 0].set_xlabel('Platform')
    axs[0, 0].set_ylim(0, 1)

    # Cosine Similarity Between Real and Synthetic Data
    cosine_sims_between = [real_synth_cosine_sim[platform] if real_synth_cosine_sim[platform] is not None else 0 for platform in platforms]
    sns.barplot(x=platforms, y=cosine_sims_between, palette=palette, ax=axs[0, 1])
    axs[0, 1].set_title('Average Cosine Similarity Between Real and Synthetic Data')
    axs[0, 1].set_ylabel('Average Cosine Similarity')
    axs[0, 1].set_xlabel('Platform')
    axs[0, 1].set_ylim(0, 1)

    # Sample Distance Within Synthetic Data
    sample_distances_within = [sample_distance_within[platform] if sample_distance_within[platform] is not None else 0 for platform in platforms]
    sns.barplot(x=platforms, y=sample_distances_within, palette=palette, ax=axs[1, 0])
    axs[1, 0].set_title('Average Sample Distance Within Synthetic Data')
    axs[1, 0].set_ylabel('Average Distance')
    axs[1, 0].set_xlabel('Platform')

    # Sample Distance Between Real and Synthetic Data
    sample_distances_between = [sample_distance_between[platform] if sample_distance_between[platform] is not None else 0 for platform in platforms]
    sns.barplot(x=platforms, y=sample_distances_between, palette=palette, ax=axs[1, 1])
    axs[1, 1].set_title('Average Sample Distance Between Real and Synthetic Data')
    axs[1, 1].set_ylabel('Average Distance')
    axs[1, 1].set_xlabel('Platform')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("cosine_similarity_and_sample_distance.png")
    plt.show()

# -------------------------
# Main Evaluation Function
# -------------------------

def evaluate_synthetic_dataset():
    """
    Main function to evaluate synthetic dataset using Vocabulary Statistics, Cosine Similarity, and Sample Distance.
    """
    # Step 1: Load Datasets
    real_data, synthetic_data = load_datasets(REAL_DATA_PATHS, SYNTHETIC_DATA_PATHS)
    if real_data.empty:
        print("Warning: Real dataset is empty.")
    if synthetic_data.empty:
        print("Warning: Synthetic dataset is empty.")
    
    # Step 2: Identify Synthetic Data
    synthetic_flags = identify_synthetic(real_data, synthetic_data, PLATFORMS)
    
    # Step 3: Compute Vocabulary Statistics
    vocab_stats = compute_vocabulary_statistics(synthetic_data, synthetic_flags, PLATFORMS, ngram=NGRAM)
    
    # Step 4: Compute Cosine Similarity Within Synthetic Data
    cosine_sim_within = compute_cosine_similarity_within_synthetic(synthetic_data, synthetic_flags, PLATFORMS, model_name=EMBEDDING_MODEL)
    
    # Step 5: Compute Cosine Similarity Between Real and Synthetic Data
    real_synth_cosine_sim = compute_cosine_similarity_between_real_synthetic(real_data, synthetic_data, PLATFORMS, model_name=EMBEDDING_MODEL)
    
    # Step 6: Compute Sample Distance Within and Between Real and Synthetic Data
    sample_distance_within = compute_sample_distance_between_real_synthetic(synthetic_data, synthetic_data, PLATFORMS, distance_metric='euclidean')
    sample_distance_between = compute_sample_distance_between_real_synthetic(real_data, synthetic_data, PLATFORMS, distance_metric='euclidean')
    
    # Step 7: Print Results
    print("\nEvaluation Results:\n")
    for platform in PLATFORMS:
        print(f"Platform: {platform}")
        print(f"  Vocabulary Size: {vocab_stats[platform]['vocab_size']}")
        print(f"  Average Cosine Similarity Within Synthetic: {cosine_sim_within[platform]:.4f}")
        print(f"  Average Cosine Similarity Between Real and Synthetic: {real_synth_cosine_sim[platform]:.4f}")
        print(f"  Average Sample Distance Within Synthetic: {sample_distance_within[platform]:.4f}")
        print(f"  Average Sample Distance Between Real and Synthetic: {sample_distance_between[platform]:.4f}\n")
    
    # Step 8: Visualization
    visualize_cosine_similarity_and_sample_distance(cosine_sim_within, real_synth_cosine_sim, sample_distance_within, sample_distance_between, PLATFORMS)
    
    # Step 9: Save Evaluation Report
    save_evaluation_report(vocab_stats, cosine_sim_within, real_synth_cosine_sim, sample_distance_between, PLATFORMS, REPORT_PATH)

if __name__ == "__main__":
    evaluate_synthetic_dataset()
