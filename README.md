# Reprogramming_LLMs_for_Multimodal_Time_Series_Forecasting
Human pose prediction plays a vital role in applications such as human-computer interaction, robot control, and intelligent surveillance. Traditional methods primarily rely on single-modal inputs like skeletal keypoints or RGB video, which often fall short in comprehensively understanding complex human motions. This thesis proposes a novel multimodal Time-LLM framework that fuses structured skeletal data (in CSV format) with monocular video information for future pose prediction.
Building upon the Time-LLM architecture, we introduce a visual patch reprogramming module that maps spatiotemporal visual features into discrete embedding representations understandable by large language models (LLMs). These visual embeddings are concatenated with prompt tokens and skeletal time-series embeddings to form a unified input sequence for the LLM, enabling cross-modal joint reasoning.
This method fully leverages the strengths of pre-trained LLMs in temporal modeling and generalization, demonstrating powerful representational and reasoning capabilities on non-linguistic sequence prediction tasks—without the need for fine-tuning. We propose exploring the framework’s potential applications in broader domains such as self-driving, multimodal biometrics, security surveillance, and abnormal behavior detection as part of future work. These directions remain conceptual and are not experimentally validated in this thesis.

1. When loading data, load both CSV data and video feature `.pt` files simultaneously:
   - CSV Data: Directly read multivariate features (e.g., 75 dimensions) excluding date and standardize them;
   - Video Features: Load from the file "video-2_features.pt", shape [T, 2048]; use a learnable linear projection to reduce to 75 dimensions, then standardize.
     Each row of 75-dimensional data is treated as an independent variable.
2. When feeding input to the frozen LLM, concatenate the encoded prompt, video (pt) encoding, and CSV encoding along the sequence dimension;
   During output, retain only the CSV encoding part and discard prompt and video (pt) parts (similar to discarding prompt in CSV encoding).
3. The model includes two PatchEmbedding modules to handle CSV and pt inputs respectively; during forecasting, prompt, pt, and CSV encodings are concatenated,
   and after LLM output, only the CSV part is sliced out as the prediction result.
4. Training logs are recorded in a `.txt` file; each print message is simultaneously written to the log file; no files are deleted after training ends.
