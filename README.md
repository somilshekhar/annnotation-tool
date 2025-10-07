# AI-Based Text Annotation App

This project is a **web-based application** for **text sentiment analysis** using a **fine-tuned BERT model**. The app predicts whether a given text is **Positive (Label 1)** or **Negative (Label 0)** and flags low-confidence predictions for potential human review.

---

## Features

* Analyze single or multiple text inputs at once.
* Uses a **custom fine-tuned BERT model** created and trained by the developer.
* Predicts sentiment with confidence scores.
* Flags low-confidence predictions (`< 0.6` by default) for human review.
* Provides a clear explanation of **Positive** and **Negative** sentiment labels in the sidebar.

---

## Sentiment Labels

* ✅ **Positive (Label 1):** The text expresses favorable, happy, or approving sentiment.
  Example: "I loved the movie", "Amazing product", "Had a wonderful experience".

* ❌ **Negative (Label 0):** The text expresses unfavorable, unhappy, or critical sentiment.
  Example: "I hated the movie", "Terrible service", "Very disappointed with the product".

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/somilshekhar/annotation-Tool.git
cd annotation-Tool

```

2. Create a Python virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

* **Windows:**

```bash
.venv\Scripts\activate
```

* **Linux/Mac:**

```bash
source .venv/bin/activate
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

1. Place your fine-tuned BERT model folder (created by you) in the project directory.

   * Folder structure example: `./fine_tuned_bert_model_hf/`
   * Make sure the folder contains:

     * `pytorch_model.bin` (model weights)
     * `config.json` (model config)
     * `tokenizer_config.json` & vocab files

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the displayed local URL in your browser to access the app.

---

## Usage

* Enter text into the input area (one line per text entry).
* Click **Analyze Text** to get sentiment predictions and confidence scores.
* Low-confidence predictions will be flagged for human review.
* Check the sidebar for **About**, **Sentiment Labels**, and flagging information.

---

## Technologies Used

* Python 3.x
* Streamlit
* PyTorch
* Hugging Face Transformers (BERT)
* Custom fine-tuned BERT model created by the developer

---

## Notes

* The app uses the Hugging Face `from_pretrained()` method to load your trained model and tokenizer. This ensures compatibility and robustness compared to pickled models.
* Make sure the model directory is correctly placed for the app to load successfully.

---

## Author

**Somil Shekhar** – Developed and fine-tuned the BERT model used in this project.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
