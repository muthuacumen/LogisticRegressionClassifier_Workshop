# Logistic Regression Classifier — Workshop

A hands-on Jupyter Notebook series that teaches **Logistic Regression Classification** from first principles, progressing from a linear regression recap through to a fully trained binary classifier — evaluated with log-loss (cross-entropy).

---

## Repository Contents

| File | Description |
| :--- | :--- |
| `LogisticRegressionClassifier_Workshop.ipynb` | Core workshop notebook — concepts, theory, and code |
| `BankLoanRepayment_LogisticRegression.ipynb` | Applied case study — predicting bank loan repayment |
| `requirements.txt` | Python package dependencies |

---

## Featured Notebook: Bank Loan Repayment Prediction

### `BankLoanRepayment_LogisticRegression.ipynb`

This notebook builds a complete binary classification pipeline around one concrete, real-world question:

> **"Given a customer's annual income, will they repay their bank loan — or default on it?"**

Every concept is anchored to this scenario, making the progression from theory to working model easy to follow.

---

### What Was Accomplished

#### 1. Demonstrated Why Linear Regression Falls Short for Classification
Linear regression predicts continuous values and is unconstrained — it can output probabilities below 0 or above 1, which is mathematically invalid for a binary outcome. The notebook visualises this failure directly on the loan dataset, motivating the need for a better approach.

#### 2. Introduced the Sigmoid (Logistic) Function
The sigmoid function is the engine behind logistic regression. It maps any real-valued input to a value strictly between 0 and 1, making it a natural fit for probability estimation. The notebook plots the S-curve alongside annotated examples showing what different probability values mean in the context of loan applicants.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

#### 3. Explained Statistical Classification and the Decision Boundary
The notebook defines core classification vocabulary (classifier, features, classes, decision boundary, training data) in plain language, then visualises a 2D decision boundary separating loan defaulters from repayers — including a live example of classifying a new applicant.

#### 4. Trained a Full Logistic Regression Classifier with scikit-learn
Using `sklearn.linear_model.LogisticRegression`, the notebook:
- Fits the model on historical income and loan outcome data
- Outputs per-customer repayment probabilities
- Applies a 0.5 decision threshold to produce final approve/reject decisions
- Reports overall model accuracy
- Demonstrates real-time prediction for a new applicant

#### 5. Implemented and Visualised Log-Loss (Cross-Entropy Loss)
Log-loss is the standard loss function for probabilistic classifiers. The notebook:
- Derives the formula from first principles
- Shows why the function heavily penalises confident wrong predictions
- Calculates per-customer loss values and visualises them as a bar chart
- Compares the model's log-loss against the random-guessing baseline (~0.693)

$$\text{Log-Loss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i)\right]$$

#### 6. Tied Everything Together in a Three-Panel Summary
A final summary chart displays the raw training data, the fitted sigmoid curve with decision boundary, and the per-customer loss profile — giving a complete end-to-end view of the model pipeline in a single figure.

---

### Key Takeaways

| Concept | One-Line Summary |
| :--- | :--- |
| **Logistic vs. Linear** | Linear regression predicts numbers; logistic regression predicts probabilities |
| **Sigmoid Function** | Squishes any number into (0, 1) — always a valid probability |
| **Decision Boundary** | The income threshold where the model switches from "reject" to "approve" |
| **Log-Loss** | Rewards confident correct predictions; severely penalises confident wrong ones |
| **Gradient Descent** | The iterative optimisation process that minimises log-loss during training |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/muthuacumen/LogisticRegressionClassifier_Workshop.git
cd LogisticRegressionClassifier_Workshop

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Open either notebook from the Jupyter interface in your browser.

---

## Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `numpy` | 2.3.0 | Numerical computing and array operations |
| `matplotlib` | 3.10.3 | Data visualisation and plotting |
| `scikit-learn` | 1.7.0 | Machine learning models and evaluation metrics |

---

## Learning Path

```
LogisticRegressionClassifier_Workshop.ipynb
  │
  ├── 0. Linear Regression Recap (MSE, R²)
  ├── 1. From Linear to Logistic Regression
  ├── 2. Statistical Classification & Decision Boundaries
  ├── 3. Classifier Implementation with scikit-learn
  └── 4. Log-Loss (Cross-Entropy) — Theory & Visualisation

BankLoanRepayment_LogisticRegression.ipynb   ← Start here for the applied example
  │
  ├── Why Linear Regression Fails for Yes/No Decisions
  ├── The Sigmoid Function — Intuition & Plots
  ├── Classification Terminology & Decision Boundaries
  ├── End-to-End Classifier (training → prediction → evaluation)
  ├── Log-Loss — Customer-by-Customer Breakdown
  └── 3 Talking Points for Class Presentation
```

---

## License

This project is intended for educational use.
