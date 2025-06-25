# Reprogramming_LLMs_for_Multimodal_Time_Series_Forecasting
1. When loading data, load both CSV data and video feature `.pt` files simultaneously:
   - CSV Data: Directly read multivariate features (e.g., 75 dimensions) excluding date and standardize them;
   - Video Features: Load from the file "video-2_features.pt", shape [T, 2048]; use a learnable linear projection to reduce to 75 dimensions, then standardize.
     Each row of 75-dimensional data is treated as an independent variable.
2. When feeding input to the frozen LLM, concatenate the encoded prompt, video (pt) encoding, and CSV encoding along the sequence dimension;
   During output, retain only the CSV encoding part and discard prompt and video (pt) parts (similar to discarding prompt in CSV encoding).
3. The model includes two PatchEmbedding modules to handle CSV and pt inputs respectively; during forecasting, prompt, pt, and CSV encodings are concatenated,
   and after LLM output, only the CSV part is sliced out as the prediction result.
4. Training logs are recorded in a `.txt` file; each print message is simultaneously written to the log file; no files are deleted after training ends.
