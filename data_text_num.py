import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
import random
from googletrans import Translator
import time

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Text augmentation functions
def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = []
        for syn in wordnet.synsets(random_word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(set(synonyms)))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def back_translation(text, src='en', tmp='fr', max_retries=3):
    translator = Translator()
    for _ in range(max_retries):
        try:
            tmp_text = translator.translate(text, src=src, dest=tmp).text
            back_translated = translator.translate(tmp_text, src=tmp, dest=src).text
            return back_translated
        except Exception as e:
            print(f"Translation error: {e}. Retrying...")
            time.sleep(1)
    return text

def random_insertion(text, n=1):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        add_word = random.choice(words)
        random_index = random.randint(0, len(new_words))
        new_words.insert(random_index, add_word)
    return ' '.join(new_words)

def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 1:
        return text
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

# Numerical augmentation functions
def add_noise(value, noise_level=0.05):
    return value + np.random.normal(0, noise_level * abs(value))

def scale(value, factor_range=(0.9, 1.1)):
    return value * np.random.uniform(*factor_range)

# Function to determine if a column is numeric
def is_numeric(column):
    return pd.api.types.is_numeric_dtype(column)

# Function to round numeric values
def round_numeric(value, original_value):
    if isinstance(original_value, int):
        return round(value)
    elif isinstance(original_value, float):
        decimal_places = len(str(original_value).split('.')[-1])
        return round(value, decimal_places)
    return value


# Function to apply text augmentation
def apply_text_augmentation(text):
    augmented_texts = [
        synonym_replacement(text, n=2),
        back_translation(text),
        random_insertion(text, n=2),
        random_deletion(text, p=0.1)
    ]
    return augmented_texts

# Function to apply numerical augmentation
def apply_numerical_augmentation(value):
    augmented_values = [
        add_noise(value),
        scale(value)
    ]
    return [round_numeric(v, value) for v in augmented_values]

# Main augmentation function
def augment_data(df, num_augmentations=20):
    augmented_data = []

    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")
        
        # Original data
        augmented_data.append(row.to_dict())

        # Augmented data
        for _ in range(num_augmentations):
            augmented_row = {}
            for column, value in row.items():
                if is_numeric(df[column]):
                    augmented_row[column] = random.choice(apply_numerical_augmentation(value))
                else:
                    augmented_row[column] = random.choice(apply_text_augmentation(str(value)))
            augmented_data.append(augmented_row)

        time.sleep(0.1)  # Small delay to avoid API rate limits

    return pd.DataFrame(augmented_data)

# Main execution
if __name__ == "__main__":
    # Load your dataset here
    # For demonstration, we'll create a sample dataset
    sample_data = {
        'text_column': ['The cat jumps.', 'A dog runs.', 'Birds fly high.'],
        'int_column': [10, 20, 30],
        'float_column': [10.5, 20.25, 30.75],
        'mixed_column': ['Text 100', 'Number 200', 'Mixed 300.50']
    }
    df = pd.DataFrame(sample_data)

    print("Original dataset:")
    print(df)
    print("\nShape of original dataset:", df.shape)

    # Save original dataset
    original_filename = 'original_version_one.csv'
    df.to_csv(original_filename, index=False)
    print(f"\nOriginal dataset saved to {original_filename}")

    # Augment data
    augmented_df = augment_data(df)

    print("\nAugmented dataset:")
    print(augmented_df)
    print("\nShape of augmented dataset:", augmented_df.shape)

    # Save augmented dataset
    augmented_filename = 'augmented_version_one.csv'
    augmented_df.to_csv(augmented_filename, index=False)
    print(f"\nAugmented dataset saved to {augmented_filename}")