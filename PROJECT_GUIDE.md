# BigMart Sales Forecasting — Full Project Walkthrough

This document explains **everything** about this project in plain language —
what it does, why every tool was chosen, every step taken to build it, every
bug hit along the way and how it was fixed, what the evaluation numbers mean,
and exactly how to run the whole thing yourself. No prior AI/ML knowledge
assumed.

---

## 1. What is this project, in one paragraph?

A supermarket chain (BigMart) has sales data for thousands of products across
several stores. We built a system that **predicts how much a given product
will sell** at a given store, based on facts about the product (weight, price,
type...) and the store (size, location, type...). But the real point of this
project isn't just "make one prediction" — it's to demonstrate **MLOps**: all
the plumbing around a machine learning model that makes it trustworthy,
repeatable, deployable, and monitorable in the real world, rather than a
one-off script in a notebook.

---

## 2. What is "MLOps"? (for someone who doesn't know AI)

Think of a regular software app: you don't just write code once — you test
it, put it in version control (Git), deploy it, and watch it in production for
problems. **MLOps applies that same discipline to machine learning models.**

A model is just a mathematical function that turns inputs (product weight,
price, etc.) into an output (predicted sales). But getting a *trustworthy*
model into production requires answering questions like:

- Where does the training data come from, and how do we know it's not broken?
- How do we compare different modeling approaches fairly?
- How do we keep track of *which* trained model is currently "the one we
  trust" (since training the same code twice can produce slightly different
  models)?
- How do we let other software (a website, an app) actually ask the model for
  a prediction?
- How do we notice if the model starts seeing very different data than what
  it was trained on (which would make its predictions unreliable)?
- How do we make sure a code change doesn't silently break something?

Every tool in this project exists to answer one of those questions.

---

## 3. The dataset

**Source**: [BigMart Sales Data on Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)
— free, no payment required, just a free Kaggle account.

**Size**: 8,523 rows. Each row = one product, at one store, with its total
sales figure.

**Columns** (raw, before we touched anything):

| Column | Meaning | Data quality issue found |
|---|---|---|
| `Item_Identifier` | Product code, e.g. `FDA15` | High-cardinality (1,559 unique values) — not useful as a raw feature |
| `Item_Weight` | Weight in kg | **1,463 missing values** |
| `Item_Fat_Content` | Low Fat / Regular | **Messy**: `Low Fat`, `low fat`, `LF`, `Regular`, `reg` all mean only 2 real categories |
| `Item_Visibility` | % of store display space given to the item | Contains **impossible zeros** (an item on sale physically occupies *some* space) |
| `Item_Type` | Category, e.g. Dairy, Snack Foods (16 types) | Clean |
| `Item_MRP` | Maximum retail price | Clean |
| `Outlet_Identifier` | Store code | 10 unique stores — same issue as Item_Identifier |
| `Outlet_Establishment_Year` | Year store opened | Clean, but more useful as "age" |
| `Outlet_Size` | Small / Medium / High | **2,410 missing values** |
| `Outlet_Location_Type` | Tier 1/2/3 city | Clean |
| `Outlet_Type` | Grocery Store / Supermarket Type1-3 | Clean |
| `Item_Outlet_Sales` | **The target** — what we're predicting | Clean |

Understanding these quality issues *before* touching any modeling code is the
first real step of any ML project — garbage in, garbage out.

---

## 4. Tools used, and why each one exists

| Tool | Role | Why this tool, in plain terms |
|---|---|---|
| **Kaggle + `kagglehub`** | Get the free dataset | Kaggle hosts thousands of free datasets; `kagglehub` is a tiny Python library that downloads them with one line of code instead of manual browser clicking |
| **pandas / numpy** | Data manipulation | The standard way to load, clean, and reshape tabular data in Python |
| **pandera** | Data validation | Like a "spell-checker" for data — it checks the incoming data matches the shape/types/ranges we expect *before* we waste time training on it. Catches broken data pipelines early |
| **scikit-learn** | Model building blocks + 2 of the 3 models | The standard, free ML library — provides the preprocessing pipeline (encoding categories, scaling numbers) and the Linear Regression / Random Forest models |
| **XGBoost** | 3rd candidate model | A very strong, industry-standard algorithm for tabular data ("gradient boosted trees") — often beats simpler models |
| **MLflow** | Experiment tracking + model registry | Every time we train, MLflow records: which code/settings were used, what the accuracy was, and saves the trained model file itself. Later, it lets us say "use whichever model we've marked as `production`" without hardcoding a filename |
| **FastAPI + Uvicorn** | Serve the model as a web API | Turns the trained model into an HTTP endpoint (`/predict`) that any other program (a website, a mobile app, another service) can call over the internet/network |
| **Docker** | Packaging | Bundles the code + all its dependencies + the trained model into one portable "box" that runs identically on any machine — solves "it works on my machine" |
| **DVC (Data Version Control)** | Pipeline orchestration + data versioning | Like Git, but for data and multi-step pipelines. Lets us run the *entire* download→train→evaluate chain with one command (`dvc repro`), and only re-runs the steps whose inputs actually changed |
| **Evidently** | Data drift detection | Compares "the data the model was trained on" vs. "the data it's seeing now" and flags if they've drifted apart statistically — an early warning that the model may need retraining |
| **Streamlit** | Monitoring dashboard | A quick way to build a web dashboard in pure Python (no HTML/JS needed) to visualize model metrics, drift reports, and let a human try predictions interactively |
| **pytest** | Automated testing | Runs small automatic checks ("does the cleaning function remove all missing values?") every time code changes, catching bugs before they reach production |
| **GitHub Actions** | CI (Continuous Integration) | Automatically runs the test suite every time code is pushed to GitHub, so broken code can't silently merge |

**Everything above is free** — no paid tiers, no cloud storage bills, no API
costs. MLflow, DVC's storage, and the dashboard all run on your own machine.

---

## 5. Step-by-step: what was actually built, in order

### Step 1 — Getting the data (`src/data/download_data.py`)
A small script that calls `kagglehub.dataset_download(...)` to pull the raw
CSV files from Kaggle into `data/raw/`. This requires a free Kaggle API
token — Kaggle's newer accounts issue a token like `KGAT_...`, which we saved
as plain text at `~/.kaggle/access_token` (older accounts get a
`kaggle.json` file instead — both are supported).

### Step 2 — Validating the data (`src/data/validate_data.py`)
Before trusting the data, we define a **schema** with `pandera`: what type each
column should be, what range of values is acceptable, which categories are
allowed. We also set "null-rate ceilings" — e.g. if more than 30% of
`Item_Weight` were suddenly missing (vs. the normal ~17%), the script *raises
an error and stops* rather than silently training on broken data. This is the
project's "responsible AI" guardrail — catching bad data before it becomes a
bad model.

### Step 3 — Feature engineering (`src/features/build_features.py`)
This is where the messy raw data gets cleaned up:
- Collapses `Low Fat` / `low fat` / `LF` into one category, same for `Regular` / `reg`.
- Derives a new `Item_Category` (Food / Drinks / Non-Consumable) from the
  first 2 letters of the product code (`FD`/`DR`/`NC`) — this is more useful
  to the model than the raw 1,559-way product code.
- Relabels fat content as `Non-Edible` for household/hygiene items, since
  "fat content" is meaningless for a product like a broom.
- Fills in missing `Item_Weight` using the average weight of that *same
  product* at other stores (same product should weigh the same everywhere).
- Fixes the impossible zero-visibility values by treating them as missing and
  filling with the average visibility for that product type.
- Fills missing `Outlet_Size` using the most common size for that outlet type.
- Converts `Outlet_Establishment_Year` (e.g. 1999) into `Outlet_Age` (e.g. 14
  years), which is more directly useful to a model than a raw year.
- Drops the high-cardinality ID columns (`Item_Identifier`, `Outlet_Identifier`)
  since they'd cause the model to "memorize" instead of generalize.

### Step 4 — Training (`src/models/train.py`)
We don't just train one model — we train **three** and compare them fairly:
Linear Regression (the simplest baseline), Random Forest, and XGBoost. Each
one is wrapped in a single scikit-learn `Pipeline` object that bundles
*both* the preprocessing (scaling numbers, one-hot-encoding categories) *and*
the model — this means the exact same object can be saved and reused for
predictions later without re-writing any encoding logic. Every run's settings
and scores are logged to MLflow automatically.

### Step 5 — Evaluation & registry (`src/models/evaluate.py`)
Looks at all the training runs MLflow recorded, picks the one with the best
accuracy (lowest RMSE — see metrics section below), and formally
**registers** it in the MLflow Model Registry under the name
`bigmart-sales-model`, tagging it with the alias `@production`. From now on,
any other part of the system that needs "the current best model" asks for
`bigmart-sales-model@production` instead of hunting for a file — so
retraining and promoting a *better* model later requires zero changes to the
serving code.

### Step 6 — Serving (`src/api/main.py`, `src/api/schemas.py`)
A FastAPI web server that loads the `@production` model once at startup, and
exposes:
- `GET /health` — is the model loaded, and which version?
- `POST /predict` — send product/store attributes as JSON, get back a
  predicted sales number (plus which model version answered, and how long the
  prediction took).

Input validation happens automatically via Pydantic — e.g. `Outlet_Type` must
be one of the 4 real categories, or the API rejects the request with a clear
error instead of silently producing a nonsense prediction.

### Step 7 — Docker
The `Dockerfile` packages the code and — importantly — **trains the model
during the image build itself**, rather than copying in an already-trained
model. (See the "bugs fixed" section below for why.) The result: `docker run`
on any machine gives an identical, self-contained, working API.

### Step 8 — DVC pipeline (`dvc.yaml`)
Defines the 5 steps (download → validate → build_features → train →
evaluate) as a dependency graph. Running `dvc repro` re-runs only the stages
whose inputs (code or data) actually changed since last time — e.g. if you
only edit `build_features.py`, DVC knows it doesn't need to re-download data,
but *does* need to re-run features, training, and evaluation.

### Step 9 — Drift monitoring (`src/monitoring/drift_report.py`)
Splits the training data into two halves, treats one as "what the model
learned from" and the other as "new incoming data," and uses Evidently to
generate an HTML report showing whether the two look statistically similar.
In a real production system, the "current" half would instead be actual new
incoming requests — this project simulates that with a held-out split so the
monitoring machinery is demonstrated end-to-end.

### Step 10 — Dashboard (`dashboard/app.py`)
A Streamlit page showing: the current production model's accuracy metrics, a
button to regenerate and view the drift report inline, and a live form where
a human can fill in product/store details and get an instant prediction —
all without touching the command line.

### Step 11 — Tests (`tests/`)
15 automated tests covering feature engineering, the training pipeline, and
the API. They're deliberately written to **not** need Kaggle access or a
trained model (the API tests fake/"stub" the model), so they can run
anywhere, including in CI, in seconds.

### Step 12 — CI (`.github/workflows/ci.yml`)
A GitHub Actions workflow that, on every push, installs dependencies and runs
the full test suite automatically.

---

## 6. Evaluation metrics — what do they actually mean?

We're predicting a *number* (sales amount), not a yes/no category, so this is
a **regression** problem. Three standard metrics judge how close predictions
are to reality:

| Metric | Plain-language meaning | Lower/Higher is better? |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | "On average, how far off are our predictions, in the same units as sales?" Large mistakes are penalized extra heavily | Lower is better |
| **MAE** (Mean Absolute Error) | "On average, how far off are our predictions?" — treats all mistakes equally, easier to interpret directly | Lower is better |
| **R²** (R-squared) | "What fraction of the pattern in sales did the model actually capture?" 1.0 = perfect, 0.0 = no better than guessing the average every time | Higher is better (0 to 1) |

**Actual results from this project** (holdout test set, i.e. data the model
never saw during training):

| Model | Cross-val RMSE | Holdout RMSE | Holdout MAE | Holdout R² |
|---|---|---|---|---|
| Linear Regression | 1148.54 | 1069.92 | 792.55 | 0.579 |
| **Random Forest** ✅ | **1105.70** | **1025.21** | **716.29** | **0.613** |
| XGBoost | 1132.29 | 1046.46 | 730.14 | 0.597 |

**Random Forest won** on every metric and was automatically registered as the
production model. In plain terms: on average, its sales predictions are off
by about ₹716–1025 (MAE/RMSE), and it explains about 61% of the variation in
sales across products and stores — a reasonable result for a first-pass model
on this dataset without heavy hyperparameter tuning.

---

## 7. Problems hit during development, and how they were fixed

Building this wasn't a straight line — here's the honest debugging log, kept
because it's genuinely informative about how real MLOps work goes:

1. **`pd.NA` crashed a type conversion.** When flagging impossible
   zero-visibility values as missing, `pd.NA` doesn't play well with
   `.astype(float)`. Fixed by using `numpy.nan` instead.

2. **MLflow misread a Windows path as a URL scheme.** `E:\MLOPS-01\mlruns`
   was parsed as scheme `"e"` (because of the colon after the drive letter),
   crashing MLflow's tracking setup. Fixed by properly converting the path to
   a `file://` URI.

3. **MLflow's file-based storage refused to start.** The installed MLflow
   version (3.15) has put its plain-folder tracking backend into
   "maintenance mode" and blocks it by default. Fixed by switching to a local
   SQLite database (`sqlite:///mlflow_store/mlflow.db`) — which is also the
   only backend that supports the full Model Registry (the alias/`@production`
   feature depends on it).

4. **XGBoost model failed to save ("untrusted type").** MLflow's newer model
   serializer (`skops`) refuses to save object types it doesn't explicitly
   trust, and XGBoost's internal `Booster` type wasn't on the default
   allow-list. Fixed by explicitly telling MLflow to trust
   `xgboost.core.Booster` when saving.

5. **Installing all dependencies at once took forever.** `pip` was stuck
   "backtracking" (silently trying different version combinations) for over
   10 minutes trying to satisfy `mlflow` + `evidently` + `dvc` + `streamlit`
   together. Fixed by installing them in smaller batches, which resolved in
   seconds each.

6. **Docker Desktop wasn't running.** The build failed instantly with a
   connection error. Fixed by starting Docker Desktop and waiting for its
   background engine to come up before retrying.

7. **The trained model wouldn't have worked inside Docker.** This was the
   most important bug: MLflow records the *absolute file path* of where a
   model's files live (e.g. `file:///E:/MLOPS-01/mlruns/...`). That path is
   meaningless inside a Linux container, which has no `E:` drive. Simply
   copying the host-trained `mlruns/` folder into the image would have
   produced a container that reports "healthy" but crashes on every real
   prediction. **Fixed by training the model *during* the Docker image build**
   instead of reusing host-trained artifacts — so the paths MLflow records are
   always native to wherever the training actually happened.

8. **A test caught a real edge-case bug.** The visibility-imputation logic
   filled missing values using the *average visibility for that product
   type* — but if an entire product type had no valid (non-zero) visibility
   values at all, there was nothing to average, leaving it still missing.
   This didn't show up on the real 8,523-row dataset (every type has plenty
   of rows), but a small 2-row unit test exposed it immediately. Fixed by
   adding a final fallback to the overall dataset average.

This list matters because it's the actual substance of "MLOps engineering" —
most of the work isn't the model itself, it's making the surrounding system
robust and portable.

---

## 8. How to run the entire project, from absolute zero

Assumes Windows with Python 3.12+ and Docker Desktop already installed.

### 8.1 — Get the code and set up Python
```bash
cd e:\MLOPS-01
python -m venv .venv
source .venv/Scripts/activate        # Git Bash. Use .venv\Scripts\activate.bat for cmd.exe
pip install -r requirements.txt
```

### 8.2 — Get a free Kaggle API token
1. Create a free account at kaggle.com.
2. Go to **Settings → API → Create New Token**.
3. If you get a token starting with `KGAT_`, save it as **plain text** (no
   quotes, no JSON) in a file at `~/.kaggle/access_token`.
   If you instead get a `kaggle.json` file, save it at `~/.kaggle/kaggle.json`.

### 8.3 — Set up DVC's local storage (free — just a folder on disk)
```bash
dvc remote add -d localstorage ../mlops-01-dvc-storage
```
(Already done in this repo — only needed once, or on a fresh clone.)

### 8.4 — Run the whole ML pipeline with one command
```bash
dvc repro
```
This downloads the data, validates it, cleans/engineers features, trains all
3 models, and registers the best one — in that order. Takes a few minutes,
mostly spent training Random Forest and XGBoost.

*(Alternative: run each step by hand — see README.md for the individual
commands.)*

### 8.5 — Look at the experiment results
```bash
mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db
```
Open the printed local URL (usually http://127.0.0.1:5000) in a browser to
see every training run, its metrics, and the registered model.

### 8.6 — Run the prediction API locally
```bash
uvicorn src.api.main:app --reload
```
Then in another terminal:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Item_Weight": 9.3, "Item_Visibility": 0.016, "Item_MRP": 249.8, "Outlet_Age": 14,
  "Item_Fat_Content": "Low Fat", "Item_Type": "Dairy", "Item_Category": "Food",
  "Outlet_Size": "Medium", "Outlet_Location_Type": "Tier 1", "Outlet_Type": "Supermarket Type1"
}'
```
Or open http://127.0.0.1:8000/docs for an interactive web page to try it
without curl.

### 8.7 — Run it in Docker instead (fully self-contained)
```bash
docker build -t bigmart-sales-api .
docker run -p 8000:8000 bigmart-sales-api
```
Same `/predict` endpoint, now running from a portable container that trained
its own model during the build — nothing from your host machine is required
at runtime.

Or bring up the API *and* an MLflow UI together:
```bash
docker compose up --build
```

### 8.8 — Generate a drift report and open the dashboard
```bash
python -m src.monitoring.drift_report
streamlit run dashboard/app.py
```
The dashboard opens in your browser automatically — check model metrics,
view the drift report, and try live predictions with a form (no typing JSON
required).

### 8.9 — Run the automated tests
```bash
pytest tests/ -v
```
Should show `15 passed`. This works even without doing steps 8.2–8.4 first,
since the tests don't depend on a trained model or Kaggle access.

### 8.10 — Push code to GitHub and let CI run automatically
```bash
git add <files>
git commit -m "..."
git push
```
GitHub Actions will automatically install dependencies and run the test
suite on every push — check the "Actions" tab on the GitHub repo page.

---

## 9. Project file structure, annotated

```
MLOPS-01/
├── data/
│   ├── raw/                    # downloaded CSVs (not committed to git)
│   └── processed/              # cleaned/engineered CSV (not committed)
├── src/
│   ├── config.py                # loads params.yaml, builds the MLflow URI
│   ├── data/
│   │   ├── download_data.py     # Step 1
│   │   └── validate_data.py     # Step 2
│   ├── features/
│   │   └── build_features.py    # Step 3
│   ├── models/
│   │   ├── train.py             # Step 4
│   │   └── evaluate.py          # Step 5
│   ├── monitoring/
│   │   └── drift_report.py      # Step 9
│   └── api/
│       ├── main.py              # Step 6 — FastAPI app
│       └── schemas.py           # request/response validation
├── dashboard/
│   └── app.py                   # Step 10 — Streamlit dashboard
├── tests/                       # Step 11 — pytest suite
├── .github/workflows/ci.yml     # Step 12 — GitHub Actions
├── Dockerfile                   # Step 7
├── docker-compose.yml           # API + MLflow UI together
├── dvc.yaml                     # Step 8 — pipeline definition
├── params.yaml                  # all tunable settings, in one place
├── requirements.txt             # Python dependencies
└── README.md                    # quick-reference version of this guide
```

---

## 10. Glossary (for the non-AI reader)

- **Model** — a mathematical function, learned from data, that turns inputs
  into a prediction.
- **Training** — the process of showing the model lots of example data so it
  can learn the pattern.
- **Regression** — predicting a number (like sales), as opposed to a category.
- **Feature** — one input column the model uses (e.g. `Item_MRP`).
- **Feature engineering** — cleaning/transforming raw data into a form the
  model can learn from better.
- **Pipeline** — a fixed sequence of steps (clean → transform → predict)
  bundled as one reusable object.
- **Experiment tracking** — recording every training attempt's settings and
  results so they can be compared later.
- **Model registry** — a catalogue of trained models with version numbers and
  labels (like "this one is the current production model").
- **API / endpoint** — a URL that other software can call to get an answer
  (here: a sales prediction) over a network.
- **Container / Docker image** — a self-contained package of code +
  dependencies that runs identically anywhere.
- **Data drift** — when the data a model sees in production starts looking
  statistically different from what it was trained on, which can silently
  degrade its accuracy.
- **CI (Continuous Integration)** — automatically testing code every time it
  changes, to catch breakage early.
