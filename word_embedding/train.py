import fasttext
import numpy as np
from datasets import load_dataset
import os
import shutil

def convert_bin_to_vec(bin_path, vec_path):
    """
    Convert a FastText .bin model file to a .vec format file.
    - `bin_path`: Path to the .bin file.
    - `vec_path`: Path where the .vec file will be saved.
    """
    try:
        # Load the .bin model
        model = fasttext.load_model(bin_path)
        print(f"Loaded .bin model with dimension: {model.get_dimension()}")

        # Save to .vec file
        model.save_model(vec_path)
        print(f"Model saved in .vec format at: {vec_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        
        
def convert_vec_to_bin(bin_path, vec_path):
    """
    Convert a FastText .bin model file to a .vec format file.
    - `bin_path`: Path to the .bin file.
    - `vec_path`: Path where the .vec file will be saved.
    """
    try:
        # Load the .bin model
        model = fasttext.load_model(vec_path)
        print(f"Loaded .bin model with dimension: {model.get_dimension()}")

        # Save to .vec file
        model.save_model(bin_path)
        print(f"Model saved in .bin format at: {bin_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        
def get_and_check_model_dimensions(model_path):
    """Check the dimensions of a FastText model."""
    try:
        model = fasttext.load_model(model_path)
        
        # Get model type
        # FastText stores model type in the args attribute
        model_args = model.f.getArgs()
        
        # model_name will be 0 for CBOW and 1 for skipgram
        model_type = "cbow" if model_args.model == 0 else "skipgram"
        print(f"Model type: {model_type}")
        
        dim = model.get_dimension()
        print(f"Model path: {model_path}")
        print(f"Model dimension: {dim}")
        
        # Get a sample word vector to verify
        # Get the first word from the model's vocabulary
        words = model.get_words()
        if words:
            sample_word = words[0]
            vector = model.get_word_vector(sample_word)
            print(f"\nSample word: {sample_word}")
            print(f"Vector dimension: {len(vector)}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return dim, model_type
# =========================
# 1. Prepare and Load Data from HF Dataset
# =========================

def prepare_data_from_hf(dataset_name, text_column='text', split='train', output_file='data.txt'):
    """
    Load a dataset from Hugging Face and prepare it for FastText training.
    - `dataset_name`: The name of the HF dataset to load.
    - `split`: The split of the dataset (e.g., 'train', 'test').
    - `output_file`: File path where the prepared text data will be saved.
    """
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name, split=split)
    
    # Save the text data into a file (one sentence per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(item[text_column] + '\n')
    
    print(f"Data saved to {output_file}")

# =========================
# 2. Training/Fine-tuning the Model
# =========================
        
def train_fasttext_model(
    data_path,             # Path to training data
    model_type='skipgram', # 'skipgram' or 'cbow'
    lr=0.05,              # Learning rate
    dim=100,              # Embedding dimension
    ws=5,                 # Context window size
    epoch=5,              # Number of epochs
    word_ngram=5,         # Word n-gram
    minCount=5,           # Minimum word frequency
    minn=3,               # Min n-gram size
    maxn=6,              # Max n-gram size
    neg=5,               # Negative sampling
    thread=4,            # Number of threads
    save_path='fasttext_model.bin',  # Output model path
    pretrained_model_path=None,      # Path to pretrained model for fine-tuning
    backup_pretrained=False           # Whether to backup pretrained model
):
    """
    Train or fine-tune a FastText model.
    If pretrained_model_path is provided, use it as initialization for further training.
    Otherwise, train from scratch.
    """
    print(f"{'Continuing training from' if pretrained_model_path else 'Training'} FastText model...")
    
    if pretrained_model_path:
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}")
        
        # Backup pretrained model if requested
        if backup_pretrained:
            backup_path = pretrained_model_path + '.backup'
            shutil.copy2(pretrained_model_path, backup_path)
            print(f"Backed up pretrained model to {backup_path}")
        
        # # model_name = pretrained_model_path.split('.')[0]
        # # vec_path = model_name + '.vec'
        # convert_vec_to_bin('ar.bin', pretrained_model_path)
        # dim, model_type = get_and_check_model_dimensions('ar.bin')
        
        # # Load pretrained model to initialize the new training
        # pretrained = fasttext.load_model(pretrained_model_path)
        # pretrained_dim = pretrained.get_dimension()
        # print(f"Using pretrained model dimension: {pretrained_dim}")
        
        # Train a new model initialized with the pretrained vectors
        model = fasttext.train_unsupervised(
            input=data_path,
            model=model_type,
            lr=lr,
            dim=300,  # Use same dimension as pretrained
            wordNgrams=word_ngram,
            ws=ws,
            epoch=epoch,
            minCount=minCount,
            minn=minn,
            maxn=maxn,
            neg=neg,
            thread=thread,
            verbose=2,
            pretrainedVectors=pretrained_model_path  # Use pretrained vectors
        )
    else:
        # Train from scratch
        model = fasttext.train_unsupervised(
            input=data_path,
            model=model_type,
            lr=lr,
            dim=dim,
            wordNgrams=word_ngram,
            ws=ws,
            epoch=epoch,
            minCount=minCount,
            minn=minn,
            maxn=maxn,
            neg=neg,
            thread=thread,
            verbose=2
        )
    
    model.save_model(save_path)
    print(f"Model saved to {save_path}")
    return model


# =========================
# 3. Evaluating the Model
# =========================

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_similarity(model, word_pairs):
    """Evaluate word similarity on a list of word pairs."""
    for word1, word2 in word_pairs:
        vec1 = model.get_word_vector(word1)
        vec2 = model.get_word_vector(word2)
        similarity = cosine_similarity(vec1, vec2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

def find_similar_words(model, word, k=10):
    """Find top k similar words to the given word."""
    print(f"\nTop {k} words similar to '{word}':")
    for neighbor in model.get_nearest_neighbors(word, k):
        print(f"{neighbor[1]} (Score: {neighbor[0]:.4f})")

# =========================
# 4. Running Everything
# =========================

if __name__ == "__main__":
    # ==== Data Preparation ====
    
    version = 'ft_1'
    # dataset to use
    DATA_PATH = "atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset"
    # Path to save the preprocessed text data
    output_file = 'al_atlas_3M4_training_data.txt'  
    # column to use
    text_column='text' 
    # Do data preparation
    PREPARE_DATA = False
    # Whether to fine-tune an existing model or train from scratch
    FINE_TUNE = True
    # Path to pretrained model for fine-tuning
    PRETRAINED_MODEL_PATH = 'cc.ar.300.vec' if FINE_TUNE else None
    # Prepare the data from HF dataset
    if PREPARE_DATA:
        prepare_data_from_hf(DATA_PATH, text_column=text_column, split='train', output_file=output_file)

    # ==== Training Parameters ====
    model_type = 'skipgram'         # 'skipgram' or 'cbow'
    lr = 0.1                   # Learning rate
    dim = 304                   # Embedding size
    word_ngram = 5              # n-gram
    ws = 5                      # Window size
    epoch = 3                   # Number of epochs
    minCount = 2                # Min word frequency
    minn = 3                    # Min n-gram size
    maxn = 6                    # Max n-gram size
    neg = 10                    # Negative sampling
    thread = os.cpu_count()     # Number of CPU threads

    # where to save model
    save_path = f'fasttext_{model_type}_v{version}.bin'
    
    # ==== Train/Fine-tune the Model ====
    model = train_fasttext_model(
        output_file, model_type, lr, dim, ws, epoch,
        word_ngram, minCount, minn, maxn, neg, thread, 
        save_path, PRETRAINED_MODEL_PATH
    )

    # ==== Evaluate the Model ====
    
    # Example word pairs to test similarity
    word_pairs = [
        ('راجل', 'مرا'),
        ('رجل', 'مرا'),
        ('راجل', 'مرأة'),
        ('رجل', 'مرأة'),
        ('باك', 'مك'),
    ]

    evaluate_similarity(model, word_pairs)

    # ==== Find Similar Words ====
    find_similar_words(model, word='راجل', k=10)
    find_similar_words(model, word='رجل', k=10)
    find_similar_words(model, word='مرا', k=10)
    find_similar_words(model, word='مرأة', k=10)
    find_similar_words(model, word='مك', k=10)