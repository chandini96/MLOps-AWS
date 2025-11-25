# 📚 Complete Component Explanation

## 🎯 Overview
This MLOps pipeline consists of 5 components that work together to fetch data, preprocess it, train a model, evaluate it, and register it in SageMaker Model Registry.

---

## 1️⃣ **Data Fetcher Component** (`components/data_fetcher/data_fetch.py`)

### **Purpose:**
Downloads the raw dataset from S3 and prepares it for the next step.

### **What It Does:**
1. **Receives Input:**
   - SageMaker copies the CSV file from S3 (`s3://mlops-data01/Heart_Disease_Prediction.csv`) to `/opt/ml/processing/input`

2. **Finds the CSV File:**
   - Searches recursively in `/opt/ml/processing/input` for any `.csv` file
   - (SageMaker may extract files to subdirectories, so it searches everywhere)

3. **Loads the Data:**
   - Uses `pandas.read_csv()` to load the CSV into a DataFrame
   - Prints the shape (rows, columns) of the loaded data

4. **Saves Output:**
   - Saves the data as `fetched_data.csv` to `/opt/ml/processing/output`
   - SageMaker automatically uploads this file to S3 at: `s3://mlops-data01/mlops-pipeline/data/fetched/`

### **Key Code Sections:**
```python
# Searches for CSV file (handles SageMaker's file extraction)
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith('.csv'):
            input_file = os.path.join(root, file)
            break

# Loads and saves data
df = pd.read_csv(input_file)
df.to_csv(output_file, index=False)
```

### **Output:**
- `fetched_data.csv` → Uploaded to S3 → Used by Preprocessing step

---

## 2️⃣ **Preprocessing Component** (`components/preprocess/preprocess.py`)

### **Purpose:**
Cleans the data, handles missing values, encodes categorical features, and splits into train/test sets.

### **What It Does:**

#### **Step 1: Setup**
- Creates output directories: `/opt/ml/processing/output/train` and `/opt/ml/processing/output/test`

#### **Step 2: Load Data**
- Finds `fetched_data.csv` from the previous step (searches in `/opt/ml/processing/input`)

#### **Step 3: Identify Target Column**
- Looks for common target column names: `target`, `Target`, `Heart Disease`, `heart_disease`
- If not found, uses the last column as the target
- **Important:** This target column is identified BEFORE preprocessing to ensure it's not modified

#### **Step 4: Handle Missing Values**
- **Numeric columns:** Fills missing values with the column's mean
- **Categorical columns:** Fills missing values with the column's mode (most frequent value)

#### **Step 5: Encode Categorical Features**
- Uses `LabelEncoder` to convert text/categorical columns to numbers
- Example: `["Male", "Female"]` → `[0, 1]`

#### **Step 6: Split Data**
- Uses `train_test_split()` with 80/20 split (80% train, 20% test)
- Random seed = 42 (ensures reproducible splits)

#### **Step 7: Save Outputs**
- Saves `train.csv` to `/opt/ml/processing/output/train/`
- Saves `test.csv` to `/opt/ml/processing/output/test/`
- SageMaker uploads these to S3:
  - Train: `s3://mlops-data01/mlops-pipeline/data/train/`
  - Test: `s3://mlops-data01/mlops-pipeline/data/test/`

### **Key Code Sections:**
```python
# Handle missing values
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].mean(), inplace=True)  # Numeric → mean
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Categorical → mode

# Encode categorical features
for col in df.select_dtypes(include=["object", "category"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```

### **Output:**
- `train.csv` → Used by Training step
- `test.csv` → Used by Evaluation step

---

## 3️⃣ **Training Component** (`components/train/train.py`)

### **Purpose:**
Trains a Logistic Regression model on the preprocessed training data.

### **What It Does:**

#### **Step 1: Load Training Data**
- Finds `train.csv` in `/opt/ml/input/data/train` (SageMaker Training Job input path)
- Searches recursively (SageMaker may extract to subdirectories)

#### **Step 2: Identify Target Column**
- Looks for `target`, `Target`, or uses the last column

#### **Step 3: Split Features and Target**
- `X` = All columns except target (features)
- `y` = Target column (labels)

#### **Step 4: Convert Target to Integer**
- Converts target to integers (handles cases where it's stored as float like `0.0`, `1.0`)
- Example: `[0.0, 1.0, 0.0]` → `[0, 1, 0]`
- **Why?** Logistic Regression expects integer class labels

#### **Step 5: Train Model**
- Creates a `LogisticRegression` model with:
  - `random_state=42` (reproducible results)
  - `max_iter=1000` (maximum iterations for convergence)
- Fits the model: `model.fit(X, y)`

#### **Step 6: Save Model**
- Saves the trained model as `model.joblib` to `/opt/ml/model/`
- SageMaker automatically uploads this to S3 at: `s3://mlops-data01/mlops-pipeline/models/`

### **Key Code Sections:**
```python
# Split features and target
X = df.drop(columns=[target_col])  # Features
y = df[target_col]                  # Target

# Convert to integer for classification
y = y.astype(int)

# Train Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "/opt/ml/model/model.joblib")
```

### **Output:**
- `model.joblib` → Uploaded to S3 → Used by Evaluation and Registration steps

---

## 4️⃣ **Evaluation Component** (`components/evaluate/evaluate.py`)

### **Purpose:**
Evaluates the trained model on test data and calculates accuracy.

### **What It Does:**

#### **Step 1: Load Trained Model**
- Finds `model.joblib` in `/opt/ml/processing/model`
- Loads it using `joblib.load()`

#### **Step 2: Load Test Data**
- Finds `test.csv` in `/opt/ml/processing/input`
- Loads it into a DataFrame

#### **Step 3: Prepare Data**
- Identifies target column (`target` or last column)
- Splits into:
  - `X_test` = Features (all columns except target)
  - `y_test` = True labels (target column)
- Converts `y_test` to integers

#### **Step 4: Make Predictions**
- Uses the model to predict: `y_pred = model.predict(X_test)`
- Converts predictions to integers

#### **Step 5: Calculate Metrics**
- Calculates **accuracy**: `accuracy_score(y_test, y_pred)`
- Accuracy = (Number of correct predictions) / (Total predictions)
- Example: If 50 out of 54 predictions are correct, accuracy = 50/54 = 0.9259 (92.59%)

#### **Step 6: Save Results**
- Creates a JSON file with:
  ```json
  {
    "problem_type": "classification",
    "metrics": {
      "accuracy": 0.9259
    },
    "samples": 54
  }
  ```
- Saves to `/opt/ml/processing/output/evaluation_results.json`
- SageMaker uploads to S3: `s3://mlops-data01/mlops-pipeline/evaluation/`

### **Key Code Sections:**
```python
# Load model and test data
model = joblib.load(model_path)
df = pd.read_csv(test_path)

# Prepare data
X_test = df.drop(columns=[target_col])
y_test = df[target_col].astype(int)

# Make predictions
y_pred = model.predict(X_test).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save results
results = {
    "problem_type": "classification",
    "metrics": {"accuracy": accuracy},
    "samples": len(y_test)
}
```

### **Output:**
- `evaluation_results.json` → Used by Model Registration step to record metrics

---

## 5️⃣ **Model Registry Component** (`components/model_registry/model_registry.py`)

### **Purpose:**
Registers the trained model in SageMaker Model Registry for versioning and deployment.

### **What It Does:**

#### **Step 1: Create Model Metrics Object**
- Reads the evaluation results from the Evaluation step
- Creates a `ModelMetrics` object that SageMaker uses to track model performance

#### **Step 2: Register Model**
- Uses SageMaker's `RegisterModel` step collection
- Registers the model with:
  - **Model artifact:** The trained model from S3
  - **Model metrics:** Evaluation results (accuracy)
  - **Model package group:** `MLOpsModelPackageGroup` (groups related model versions)
  - **Approval status:** `Approved` (can be changed manually in console)
  - **Inference instances:** Types of instances that can run this model (`ml.t2.medium`, `ml.m5.xlarge`)

#### **Step 3: Model Versioning**
- Each pipeline run creates a new model version
- All versions are stored in the same model package group
- You can compare versions, promote to production, etc.

### **Key Code Sections:**
```python
# Create model metrics from evaluation results
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        content_type="application/json"
    )
)

# Register model
step_register = RegisterModel(
    name="RegisterModel",
    estimator=train_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    model_metrics=model_metrics,
    model_package_group_name="MLOpsModelPackageGroup",
    approval_status="Approved"
)
```

### **Output:**
- Model registered in SageMaker Model Registry
- Can be viewed in AWS Console → SageMaker → Model Registry
- Can be deployed to endpoints or used for batch inference

---

## 🔄 **Pipeline Orchestration** (`pipeline.py`)

### **Purpose:**
Defines the complete workflow, connects all components, and manages execution.

### **What It Does:**

#### **1. Configuration**
- Sets up AWS credentials, S3 bucket, ECR image URIs
- Defines pipeline parameters (input data path, instance types, etc.)

#### **2. Defines Each Step**
- **DataFetch:** Uses `ScriptProcessor` to run `data_fetch.py` in a Docker container
- **PreprocessData:** Uses `ScriptProcessor` to run `preprocess.py`
- **TrainModel:** Uses `Estimator` to run `train.py` (Training Job)
- **EvaluateModel:** Uses `ScriptProcessor` to run `evaluate.py`
- **RegisterModel:** Uses `RegisterModel` to register the model

#### **3. Connects Steps**
- Each step's output becomes the next step's input:
  ```
  DataFetch → PreprocessData → TrainModel → EvaluateModel → RegisterModel
  ```

#### **4. Data Flow**
```
S3 (raw data) 
  → DataFetch (fetched_data.csv)
    → PreprocessData (train.csv, test.csv)
      → TrainModel (model.joblib)
        → EvaluateModel (evaluation_results.json)
          → RegisterModel (registered model version)
```

#### **5. Execution Functions**
- `deploy_pipeline()`: Creates/updates the pipeline definition in SageMaker
- `start_execution()`: Starts a new pipeline run
- `monitor_execution()`: Polls the pipeline status and shows step progress

### **Key Code Sections:**
```python
# Define pipeline with all steps
pipeline = Pipeline(
    name="MLOpsPipeline",
    steps=[step_data_fetch, step_preprocess, step_train, step_evaluate, step_register],
    parameters=[input_data_s3_uri, processing_instance_type, training_instance_type, model_approval_status]
)

# Deploy and run
deploy_pipeline()      # Upload pipeline definition
exec_obj = start_execution()  # Start execution
monitor_execution(exec_obj.arn)  # Monitor progress
```

---

## 📊 **Complete Data Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                    S3 Bucket (mlops-data01)                     │
│  s3://mlops-data01/Heart_Disease_Prediction.csv (Raw Data)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1️⃣ DATA FETCH (Processing Job)                                │
│  • Reads CSV from S3                                            │
│  • Output: fetched_data.csv                                     │
│  • S3: s3://.../mlops-pipeline/data/fetched/                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2️⃣ PREPROCESS (Processing Job)                                │
│  • Handles missing values                                       │
│  • Encodes categorical features                                 │
│  • Splits into train/test (80/20)                              │
│  • Output: train.csv, test.csv                                 │
│  • S3: s3://.../mlops-pipeline/data/train/                     │
│  • S3: s3://.../mlops-pipeline/data/test/                      │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  3️⃣ TRAIN (Training Job)     │  │  4️⃣ EVALUATE (Processing)   │
│  • Loads train.csv           │  │  • Loads model.joblib       │
│  • Trains LogisticRegression │  │  • Loads test.csv           │
│  • Output: model.joblib      │  │  • Calculates accuracy      │
│  • S3: s3://.../models/      │  │  • Output: evaluation.json  │
└──────────────┬───────────────┘  └──────────────┬───────────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5️⃣ REGISTER MODEL (Model Registry)                             │
│  • Registers model with metrics                                 │
│  • Creates model version                                        │
│  • Stores in Model Package Group                                │
│  • Ready for deployment                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 **Key Concepts**

### **SageMaker Processing Jobs**
- Used for: Data Fetch, Preprocess, Evaluate
- Runs scripts in Docker containers
- Input/output paths are automatically mounted
- Outputs are automatically uploaded to S3

### **SageMaker Training Jobs**
- Used for: Training
- Runs training scripts in Docker containers
- Automatically uploads model artifacts to S3
- Can use GPU instances for deep learning

### **SageMaker Model Registry**
- Stores model versions
- Tracks metrics and metadata
- Enables model versioning and approval workflows
- Can deploy models to endpoints

### **Docker Containers**
- Each component runs in its own Docker container
- Images are stored in ECR (Elastic Container Registry)
- Containers include all dependencies (pandas, sklearn, etc.)

---

## 🎯 **Summary**

1. **Data Fetch:** Gets raw data from S3
2. **Preprocess:** Cleans data, encodes features, splits train/test
3. **Train:** Trains Logistic Regression model
4. **Evaluate:** Tests model and calculates accuracy
5. **Register:** Saves model to Model Registry for deployment

All steps are orchestrated by `pipeline.py`, which ensures they run in the correct order and pass data between them automatically.

