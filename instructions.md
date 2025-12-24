# Instructions to Obtain Dataset and Run Pipeline

## 1. Data Acquisition
The dataset used is the **UCI Heart Disease Dataset** (Cleveland).

### Automatic Download
The pipeline is designed to automatically acquire the dataset. No manual action is needed. 
When you run the pipeline, the system will:
1.  **Attempt to Download:** It tries to download the dataset directly from the UCI Machine Learning Repository URL:
    `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
2.  **Fallback to Local:** If the download fails (e.g., no internet), it looks for a local copy at `data/heart+disease/processed.cleveland.data`.
3.  **Process:** The raw data is cleaned, column names are added, and it is saved as `data/heart.csv` for use.

### Manual Download (Optional)
If you wish to manually download the data:
1.  Download the file from [UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).
2.  Place it in `data/heart+disease/processed.cleveland.data`.

---

## 2. Running the Pipeline

### Prerequisites
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Option A: Run Full Pipeline (EDA + Training)
To generate reports and train the model in one go:
```bash
python src/auto_eda.py && python src/train_automl.py
```

### Option B: Run Steps Individually
1.  **EDA (Exploratory Data Analysis):**
    Generates a Pandas Profiling report in `plots/pandas_profiling_report.html`.
    ```bash
    python src/auto_eda.py
    ```

2.  **Model Training (AutoML):**
    Runs FLAML to find the best model and logs results to MLflow.
    ```bash
    python src/train_automl.py
    ```

---

## 3. Viewing Results
Start the MLflow UI to view experiment logs, metrics, text reasoning, and artifacts.
```bash
mlflow ui
```
Open your browser at `http://127.0.0.1:5000`.
