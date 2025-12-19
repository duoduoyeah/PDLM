
import pickle
import tiktoken
import os
import datetime # Added for timestamp
from nanochat.tokenizer import RustBPETokenizer

# --- 1. Load the existing tokenizer ---
# Replace with the path to the directory containing your tokenizer.pkl
tokenizer_dir = "my_tokenizer_dir"
original_pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")

try:
    with open(original_pickle_path, "rb") as f:
        existing_enc = pickle.load(f)
except FileNotFoundError:
    print(f"Error: The file {original_pickle_path} was not found.")
    # Create the directory if it doesn't exist
    os.makedirs(tokenizer_dir, exist_ok=True)
    print(f"Please put your tokenizer.pkl file into the directory: {tokenizer_dir}")
    print("The script will now exit. Please run it again after placing your tokenizer file.")
    exit(1) # Quit the script


# --- 2. Define your new special tokens ---
new_special_tokens = ["<|new_token_1|>", "<|new_token_2|>"]

# --- 3. Create the new special_tokens dictionary ---
# Copy the existing special tokens
updated_special_tokens = existing_enc._special_tokens.copy()

# Add the new tokens with new IDs starting from the end of the current vocab
current_vocab_size = existing_enc.n_vocab
for token in new_special_tokens:
    if token not in updated_special_tokens:
        updated_special_tokens[token] = current_vocab_size
        current_vocab_size += 1

# --- 4. Create a new tiktoken.Encoding object ---
# This reconstructs the tokenizer with the added special tokens
new_enc = tiktoken.Encoding(
    name=existing_enc.name, # 'name' is a public attribute
    pat_str=existing_enc._pat_str, # Use the correct internal attribute _pat_str
    mergeable_ranks=existing_enc._mergeable_ranks, # Use the correct internal attribute _mergeable_ranks
    special_tokens=updated_special_tokens
)

# --- 5. Save the new tokenizer to the same directory with a timestamp ---
# Generate a timestamp for the new tokenizer file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Construct the new filename with the timestamp
new_pickle_filename = f"tokenizer_{timestamp}.pkl"
new_pickle_path = os.path.join(tokenizer_dir, new_pickle_filename) # Use original tokenizer_dir

with open(new_pickle_path, "wb") as f:
    pickle.dump(new_enc, f)

print(f"Tokenizer successfully updated and saved to {new_pickle_path}")
print(f"Original vocab size: {existing_enc.n_vocab}")
print(f"New vocab size: {new_enc.n_vocab}")

# You can now load the new tokenizer directly from the timestamped file
with open(new_pickle_path, "rb") as f:
    loaded_new_enc = pickle.load(f)
# Instantiate RustBPETokenizer with the loaded encoding and the appropriate bos_token
new_tokenizer = RustBPETokenizer(loaded_new_enc, "<|bos|>")

# Verify that the new token can be encoded
try:
    new_token_id = new_tokenizer.encode_special("<|new_token_1|>")
    print(f"Successfully encoded '<|new_token_1|>' to ID: {new_token_id}")
except KeyError:
    print("Error: Could not encode the new special token. Something went wrong.")
