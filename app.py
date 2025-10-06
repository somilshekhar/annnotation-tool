# app.py

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification # Import BertForSequenceClassification

# --- Configuration ---
# Define the path to your saved model directory (not the .pkl file)
MODEL_DIRECTORY = "./fine_tuned_bert_model_hf" # Update this path to your downloaded folder

# Define the model name used for the tokenizer (still needed for from_pretrained)
MODEL_NAME = "bert-base-uncased" # Or the specific model name you used

# Define the maximum sequence length used during training
MAX_LENGTH = 128
# Define the confidence threshold for flagging (adjust as needed)
CONFIDENCE_THRESHOLD = 0.6

# --- Load Model and Tokenizer (using from_pretrained) ---
@st.cache_resource # Cache the model and tokenizer to avoid reloading on every rerun
def load_model_and_tokenizer(model_directory):
    """Loads the model and tokenizer using from_pretrained."""
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        # Load the model, PyTorch handles mapping to CPU/GPU automatically
        model = BertForSequenceClassification.from_pretrained(model_directory)
        # Set the model to evaluation mode
        model.eval()
        return model, tokenizer
    except FileNotFoundError:
        st.error(f"Error: Model directory not found at {model_directory}. Make sure to download the '{MODEL_DIRECTORY}' folder from Colab and place it in the same directory as app.py")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer(MODEL_DIRECTORY)

# --- Prediction Function ---
def predict_text(texts):
    """
    Predicts labels and confidence scores for text inputs.
    Handles both single and batch inputs.
    """
    if model is None or tokenizer is None:
        return [{"text": text, "predicted_label": None, "confidence_scores": None, "needs_review": True} for text in texts]

    # Tokenize the input texts
    inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")

    # Ensure model is on the correct device (CPU in this case, handled by from_pretrained, but explicit is fine)
    device = torch.device('cpu')
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}


    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get confidence scores
    confidence_scores = F.softmax(logits, dim=-1).tolist()

    # Determine predicted labels
    predicted_labels = torch.argmax(logits, dim=-1).tolist()

    # Prepare the results with flagging
    results = []
    for i in range(len(texts)):
        result = {
            "text": texts[i],
            "predicted_label": predicted_labels[i],
            "confidence_scores": confidence_scores[i]
        }
        # Flag low-confidence predictions
        highest_confidence = confidence_scores[i][predicted_labels[i]]
        result['needs_review'] = highest_confidence < CONFIDENCE_THRESHOLD
        results.append(result)

    return results

# --- Streamlit App Layout ---
st.title("AI-Based Text Annotation App")
st.markdown("Enter text below to get sentiment predictions (0: Negative, 1: Positive).")

# Text input area
user_input = st.text_area("Enter text to analyze (one entry per line):", "")

# Prediction button
if st.button("Analyze Text"):
    if user_input:
        # Split input by lines to handle multiple inputs
        texts = [text.strip() for text in user_input.split('\n') if text.strip()]

        if texts:
            # Get predictions
            predictions = predict_text(texts)

            # Display results
            st.subheader("Analysis Results:")
            for result in predictions:
                st.write(f"**Text:** {result['text']}")
                if result['predicted_label'] is not None:
                    sentiment = "Positive" if result['predicted_label'] == 1 else "Negative"
                    st.write(f"**Predicted Sentiment:** {sentiment} (Label: {result['predicted_label']})")
                    st.write(f"**Confidence Scores:** {result['confidence_scores']}")
                    if result['needs_review']:
                        st.warning("Flagged for Human Review (Low Confidence)")
                else:
                     st.write("Prediction failed due to model/tokenizer loading error.")
                st.markdown("---") # Separator

        else:
            st.warning("Please enter some text to analyze.")
    else:
        st.warning("Please enter some text to analyze.")

# Information about the model and flagging
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a fine-tuned BERT model for text sentiment classification.\n\n"
    f"Predictions with a confidence score below {CONFIDENCE_THRESHOLD} are flagged for potential human review."
)

# Disclaimer about pickling (updated)
st.sidebar.info(
    "This app loads the model and tokenizer using the recommended Hugging Face `from_pretrained()` method, "
    "which is more robust and compatible than using pickle."
)
st.sidebar.info(
     "**Sentiment Labels:**\n"
    "- **Positive (Label 1):** The text expresses a favorable, happy, or approving sentiment. Examples: 'I loved the movie', 'Amazing product', 'Had a wonderful experience'.\n"
    "- **Negative (Label 0):** The text expresses an unfavorable, unhappy, or critical sentiment. Examples: 'I hated the movie', 'Terrible service', 'Very disappointed with the product'.\n\n"
    "The app also flags sentences with low confidence (< threshold) for human review, in case the model is unsure."
)