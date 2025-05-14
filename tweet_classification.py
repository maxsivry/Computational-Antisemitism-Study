import datasets
from transformers import pipeline
from tqdm.auto import tqdm
import pandas as pd
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset

# Load the model for classification
pipe = pipeline(model="mivry/antisemitism_model_jikeli")

# uncomment to use with personally annotated dataset . .
# pipe = pipeline(model="mivry/antisemitism_model")

# Load the dataset
original_df = pd.read_csv("pure_text_dr.csv")
original_df = original_df.dropna(subset=["pure_text"])
print("success")

# Creating empty columns
original_df["label"] = ""
original_df["score"] = ""
print("looping")

# Use tqdm to create a progress bar
for index, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Classifying texts"):
    # Truncate the text to fit within the model's maximum sequence length
    try:
        # Run classification on each row, appending to the dataframe
        prediction = pipe(str(row["pure_text"]))
        original_df.at[index, "label"] = prediction[0]["label"]
        original_df.at[index, "score"] = prediction[0]["score"]

    except Exception as e:
        # Handle the exception (print or log the error)
        print(f"Error processing row {index}: {e}")

# Reapplying 1s and 0s to label.
original_df["label"] = original_df["label"].apply(lambda x: 1 if x == "A" else 0)
print(original_df.head(50))
original_df.to_csv('results.csv', index=False)
print("done!")
