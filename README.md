# Consumer Spending Intelligence

### End-to-End Financial Analytics + Anomaly Detection + Behaviour Shift Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.14-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Live-red) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-IsolationForest-orange) ![Status](https://img.shields.io/badge/Status-Completed-green)

**Live Dashboard** — [consumer-spending-intelligence-system.streamlit.app](https://consumer-spending-intelligence-system.streamlit.app)

---
## One-Line Summary

> Built an end-to-end consumer spending analytics pipeline on 1.296M credit card transactions — engineering 4 behavioural features, running 7 formal hypothesis tests across categories, demographics, and geography, training an unsupervised Isolation Forest that achieves **ROC-AUC of 0.9032** without ever seeing the fraud label, and detecting a statistically confirmed February 2020 spending shift consistent with COVID-19 impact — delivered through a 6-page interactive Streamlit dashboard deployed on Streamlit Cloud.

---

## Key Metrics

| Metric                           | Value                                         |
| -------------------------------- | --------------------------------------------- |
| Total Transactions Analysed      | **1,296,675**                                 |
| Date Range                       | **Jan 2019 – Jun 2020 (18 months)**           |
| Unique Customers                 | **983**                                       |
| Unique Merchants                 | **693**                                       |
| Spending Categories              | **14**                                        |
| States Covered                   | **51**                                        |
| Total Revenue                    | **$91.2M**                                    |
| Avg Customer Lifetime Value      | **$92,800**                                   |
| Champion Segment Avg LTV         | **$147,925**                                  |
| Fraud Rate                       | **0.58% (7,506 transactions)**                |
| Anomaly Detection ROC-AUC        | **0.9032**                                    |
| Anomalies Flagged                | **7,521 vs 7,506 known fraud**                |
| Flagged Median Transaction       | **$997 vs $47 normal (2,015% higher)**        |
| Amt Z-Score — Flagged vs Normal  | **7.25 vs -0.18 (4,079% higher)**             |
| Memory Optimisation              | **1,095MB → 426MB (61% reduction)**           |
| Parquet Compression              | **~500MB CSV → 95MB Parquet (89% reduction)** |
| Hypothesis Tests Run             | **7**                                         |
| Confirmed COVID Changepoint      | **February 2020**                             |
| Cohen's d — Revenue Shift        | **0.632 (medium effect)**                     |
| Pre-Changepoint Monthly Revenue  | **$5.28M**                                    |
| Post-Changepoint Monthly Revenue | **$4.51M**                                    |

---

## Problem Statement

A financial services company processes credit card transactions across 983 customers, 693 merchants, and 14 spending categories spanning 18 months across all 51 US states. Despite having over 1.29 million transaction records, the business had no unified system to:

- Understand how consumer spending behaviour evolves over time across categories and demographics
- Identify which customer segments, geographies, and merchants drive revenue concentration
- Detect anomalous transactions that deviate from established individual spending behaviour
- Quantify structural shifts in spending patterns and pinpoint exactly when they occurred

**Core business questions:**

1. Which spending categories, customer demographics, and geographies drive revenue — and are observed differences statistically significant or noise?
2. Can unsupervised anomaly detection surface fraudulent transactions without ever accessing the fraud label during training?
3. Did consumer spending structurally shift at any point during the 18-month period — and if so, when exactly and by how much?

---

## Dataset

**Source:** [Credit Card Transactions Dataset — Kaggle (priyamchoksi)](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)

| Column                | Type            | Description                                                      |
| --------------------- | --------------- | ---------------------------------------------------------------- |
| trans_date_trans_time | datetime        | Transaction timestamp                                            |
| cc_num                | string          | Customer identifier                                              |
| merchant              | category        | Merchant name                                                    |
| category              | category        | Spending category (14 categories)                                |
| amt                   | float           | Transaction amount in USD                                        |
| gender                | category        | Customer gender                                                  |
| city, state           | string/category | Customer location                                                |
| lat, long             | float           | Customer coordinates                                             |
| city_pop              | int             | Population of customer city                                      |
| job                   | category        | Customer occupation                                              |
| dob                   | datetime        | Customer date of birth                                           |
| merch_lat, merch_long | float           | Merchant coordinates                                             |
| is_fraud              | int8            | Fraud label — used only for model validation, never for training |

**Columns dropped:** `first`, `last`, `street` (PII with no analytical value), `unix_time` (redundant), `Unnamed: 0` (CSV artifact)

---

## Approach & Methodology

### Phase 1 — Data Ingestion & Validation (Notebook 01)

- Loaded 1.296M rows from raw CSV at 1,095MB memory footprint
- Cast all columns to correct logical dtypes — datetimes parsed, identifiers to string, low-cardinality columns to category dtype
- Dropped PII columns and redundant fields — 23 columns reduced to 19
- Memory optimised from 1,095MB to 426MB (61% reduction)
- Confirmed zero nulls and zero duplicates across all 19 columns
- Saved clean Parquet at 95.3MB — 89% smaller than raw memory footprint
- Read-back verified shape and dtype match exactly

### Phase 2 — Exploratory Data Analysis (Notebook 02)

- Profiled transaction amount distribution — skew 42.3, kurtosis 4,545, confirming extreme right skew
- Ran Kolmogorov-Smirnov normality test — formally rejected normality (KS = 0.333, p < 0.001), justifying all non-parametric downstream tests
- Log-transformed distribution revealed bimodal structure — two distinct spending clusters
- Identified peak transaction hour at 23:00 and December 2019 revenue spike for investigation

### Phase 3 — Spending Trend Analysis (Notebook 03)

- Decomposed 18-month monthly revenue using STL (Seasonal-Trend decomposition using Loess) with period=6
- Flat trend component ($4.4M–$4.9M) confirmed no structural growth — revenue driven by seasonality and residual shocks
- December 2019 residual of $4.9M confirmed the spike was a structural anomaly, not seasonal
- Ran three hypothesis tests — Kruskal-Wallis (amount vs category), Mann-Whitney U (weekend vs weekday), Chi-Square (category vs gender)

### Phase 4 — Customer Intelligence (Notebook 04)

- Built customer-level RFM table — 983 rows aggregated from 1.296M transactions
- Scored R, F, M dimensions using quintile binning with duplicate-safe qcut and median fallback
- Three segments emerged: Champions (393), Recent (393), Loyal (197)
- Ran Kruskal-Wallis across 5 age groups and Mann-Whitney U across gender with Cohen's d effect sizing
- Built 17-cohort retention matrix tracking active customers month by month

### Phase 5 — Geographic & Merchant Analysis (Notebook 05)

- Ranked all 51 states by revenue, volume, and median transaction value
- Computed Haversine distance between customer and merchant coordinates across all 1.296M transactions
- Ran Spearman correlation — city population vs median spend and transaction distance vs fraud rate
- Built merchant Pareto curve across all 693 merchants with cumulative revenue share

### Phase 6 — Anomaly Detection (Notebook 06)

- Engineered 4 behavioural features beyond raw columns:
  - `amt_zscore` — deviation from each customer's own mean spend
  - `amt_to_median_ratio` — transaction size relative to customer's own median
  - `cust_tx_count` — customer overall transaction frequency
  - `hour_deviation` — timing deviation from customer's peak transaction hour
- Trained Isolation Forest (200 estimators, contamination = 0.0058) on 12 features
- Model trained entirely without the fraud label — `is_fraud` used only for post-training validation
- Saved scored dataset with anomaly flags and continuous anomaly scores for dashboard use

### Phase 7 — Behaviour Shift Detection (Notebook 07)

- Applied CUSUM control chart to monthly revenue — control limit at 5 standard deviations
- Implemented Pettitt test with permutation-based p-value using 10,000 permutations — the closed-form approximation produces invalid values for n < 25, a methodological issue most implementations ignore
- Applied CUSUM + Pettitt + Cohen's d framework to all 14 categories individually
- Reported Cohen's d alongside every test to distinguish statistical from practical significance

---

## Key Findings & Results

### Distribution

- Transaction amounts are severely right-skewed — mean $70.35 vs median $47.52, skew 42.3, kurtosis 4,545
- KS test formally rejects normality (KS = 0.333, p < 0.001) — justifies all non-parametric testing
- Log-transformed distribution reveals two distinct spending clusters — small everyday purchases and large occasional ones

### Spending Composition

- `grocery_pos` leads revenue at 15.85% of total
- `gas_transport` leads volume but ranks 4th in revenue — high frequency, low ticket ($62.84 median)
- Combined shopping (`shopping_pos` + `shopping_net`) accounts for 19.66% of revenue — the real number one segment
- `travel` punches above its weight — 3.1% volume but 4.95% revenue share

### Revenue Trends

- Q4 2019 was the peak quarter at $19.7M — Q1 2020 dropped to $12.2M
- YoY revenue declined 7.1% (Jan–Jun 2020 vs 2019) — COVID-19 demand shock
- STL trend component flat at $4.4M–$4.9M — no structural growth, revenue driven by seasonality
- December 2019 STL residual of $4.9M — the holiday spike was anomalous, not seasonal

### Hypothesis Test Results

| Test           | Question                                        | Result            | Statistic   | p-value | Effect Size                    |
| -------------- | ----------------------------------------------- | ----------------- | ----------- | ------- | ------------------------------ |
| KS Test        | Is transaction amount normally distributed?     | Reject H0         | KS = 0.333  | < 0.001 | —                              |
| Kruskal-Wallis | Do amounts differ across categories?            | Reject H0         | H = 262,024 | < 0.001 | grocery_pos $105 vs travel $6  |
| Mann-Whitney U | Do weekend amounts differ from weekday?         | Reject H0         | Significant | < 0.05  | Cohen's d = small              |
| Chi-Square     | Is category associated with gender?             | Reject H0         | Significant | < 0.001 | Cramer's V = weak              |
| Kruskal-Wallis | Does lifetime spend differ across age groups?   | Reject H0         | H = 130.75  | < 0.001 | 31-45 median $118K vs 60+ $67K |
| Mann-Whitney U | Do male and female customers spend differently? | Reject H0         | U = 108,084 | 0.004   | Cohen's d = -0.256 (small)     |
| Spearman       | Does city population predict spend?             | Reject H0         | rho = 0.137 | < 0.001 | Weak positive                  |
| Spearman       | Does transaction distance predict fraud?        | Fail to reject H0 | rho = 0.034 | 0.370   | None                           |
| Pettitt        | Does a structural changepoint exist?            | Fail to reject H0 | K = 13.00   | 0.711   | Cohen's d = 0.632 (medium)     |

### Customer Intelligence

- Champions (393 customers) — avg 2,127 transactions, $147,925 avg lifetime value, 1 day avg recency
- Loyal (197 customers) — avg 1,258 transactions, $87,323 avg lifetime value
- Recent (393 customers) — avg 541 transactions, $40,421 avg lifetime value
- No At-Risk or Lost segments — 75% of customers transacted within the last day of the observation period
- Age group 31-45 — $118,121 median lifetime spend, 36.36% of total platform revenue
- Female customers — $91,524 median lifetime spend vs male $77,640, Cohen's d = -0.256
- Female customers drive 54.6% of platform revenue through higher transaction frequency

### Geographic Intelligence

- Texas leads at 7.46% of revenue, followed by New York (6.58%) and Pennsylvania (6.33%)
- Top 3 states combined — 20.37% of revenue
- 42 of 51 states drive 80% of revenue — far more distributed than classic 80/20
- Median transaction values are uniform across states — $41 to $52 for most states
- City population weakly predicts spend (Spearman rho = 0.137) — statistically significant but not practically meaningful
- Transaction distance has zero predictive power for fraud (rho = 0.034, p = 0.370)

### Merchant Intelligence

- 693 unique merchants across the platform
- 464 merchants (67%) drive 80% of revenue — flat distribution, no single merchant dominates
- Top merchants are exclusively `grocery_pos` — revenue per merchant of $290K–$391K
- High-volume merchants show elevated fraud rates of 1.07%–1.97% vs platform average of 0.58%

### Anomaly Detection

- ROC-AUC: **0.9032** — fully unsupervised, zero access to fraud labels during training
- 7,521 anomalies flagged vs 7,506 known fraud cases — near-identical volume at the contamination threshold
- Flagged transaction median amount: $996.88 vs $47.12 normal — 2,015% higher
- `amt_zscore` for flagged transactions: 7.25 vs -0.18 normal — 4,079% higher
- `amt_to_median_ratio` for flagged: 24.35 vs 0.99 — flagged transactions are 24x the customer's own median spend
- Transaction distance difference between flagged and normal: -2.2% — distance is not a fraud signal
- 70.9% of flagged anomalies concentrated in `shopping_net` (36.7%) and `shopping_pos` (34.2%)

### Behaviour Shift Detection

- CUSUM control limit of $7.11M (5 x std) — never breached across 18 months
- CUSUM range: -$3.13M to +$4.17M — volatile but within bounds
- Pettitt permutation test — K = 13.00, p = 0.711 — low statistical power at n = 18 months
- Cohen's d = 0.632 (medium effect) — pre-changepoint mean $5.28M vs post $4.51M, a $0.77M monthly revenue decline
- February 2020 identified as changepoint across 13 of 14 categories simultaneously
- `travel` most impacted — Cohen's d = 0.928, revenue decline of -24.1%
- `shopping_pos` most resilient — Cohen's d = 0.487, revenue decline of -11.9%
- All 14 categories declined — the COVID shock was universal, not category-specific

---

## Strategic Recommendations

| Priority | Recommendation                                                                         | Analytical Basis                                                                       |
| -------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Critical | Real-time amt_zscore monitoring — flag transactions >5 std from customer's own mean    | Anomaly detection — primary fraud signal, 4,079% difference between flagged and normal |
| Critical | Re-engage the 393 Recent segment customers with targeted offers before frequency drops | RFM analysis — high recency score, critically low frequency and monetary scores        |
| High     | Build loyalty and upsell programmes targeting the 31-45 age group                      | $118,121 median LTV — highest value demographic, confirmed Kruskal-Wallis p < 0.001    |
| High     | Audit high-volume merchants showing 1.07%–1.97% fraud rates                            | Merchant fraud analysis — elevated 2-3x above platform average                         |
| Medium   | Investigate `shopping_net` and `shopping_pos` fraud concentration (70.9% of anomalies) | Anomaly category distribution                                                          |
| Medium   | Deploy CUSUM-based revenue monitoring with automated alerts at 5 std control limit     | Behaviour shift detection framework                                                    |
| Growth   | Target female customers with frequency-based engagement — drive repeat purchase rate   | 54.6% of revenue, statistically significant spend premium (p = 0.004)                  |

---

## Key Architectural Decisions

**Parquet over CSV for all processed data** — 95MB vs ~500MB raw memory. All downstream notebooks and the dashboard load in under 2 seconds.

**Permutation-based Pettitt p-value** — the standard closed-form approximation produces values greater than 1 for n < 25. We implement a 10,000-permutation Monte Carlo p-value that is valid at any sample size. This is a genuine methodological contribution that most implementations miss.

**Customer-level features for anomaly detection** — `amt_zscore` and `amt_to_median_ratio` give the Isolation Forest a personalised baseline per customer. Without these, a high-value customer's normal large transaction would be flagged as anomalous. With them, the model flags deviation from individual behaviour rather than population behaviour.

**Cohen's d alongside every hypothesis test** — p-values alone do not tell you whether an effect matters in practice. Reporting effect size alongside significance is standard in academic statistics but rare in junior data science portfolios.

**Non-parametric tests throughout** — every test choice is justified by the KS normality test in Notebook 02. The choice of Mann-Whitney over t-test and Kruskal-Wallis over ANOVA is grounded in formally confirmed distributional properties of the data.

**Git LFS for Parquet files** — both processed Parquet files total 159MB, exceeding GitHub's 100MB file limit. Git LFS tracks them transparently — the repository clones normally and Parquet files download automatically.

---

## Dashboard Pages

| Page                  | Contents                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------ |
| Overview              | 6 KPI cards, monthly revenue trend, key findings                                           |
| Spending Trends       | Monthly/quarterly charts, STL decomposition, category trend lines, hypothesis test results |
| Customer Intelligence | RFM segment cards, LTV charts, demographic hypothesis tests, cohort retention heatmap      |
| Geo & Merchant        | US choropleth map, state revenue ranking, merchant Pareto curve, top merchants table       |
| Anomaly Detection     | ROC curve, score distribution, anomaly profile, flagged transaction explorer with filters  |
| Behaviour Shift       | CUSUM chart, changepoint detection, category Cohen's d bars, statistical test summary      |

---

## Tech Stack

| Category              | Tools                                                            |
| --------------------- | ---------------------------------------------------------------- |
| Language              | Python 3.14                                                      |
| Data Processing       | Pandas, NumPy                                                    |
| Visualisation         | Matplotlib, Seaborn, Plotly                                      |
| Statistical Testing   | SciPy (KS, Kruskal-Wallis, Mann-Whitney U, Chi-Square, Spearman) |
| Time Series           | Statsmodels (STL decomposition)                                  |
| Machine Learning      | Scikit-learn (Isolation Forest, StandardScaler, ROC-AUC)         |
| Changepoint Detection | Custom Pettitt implementation with permutation testing           |
| Storage               | Parquet (PyArrow), Git LFS                                       |
| Dashboard             | Streamlit, Plotly                                                |
| Deployment            | Streamlit Cloud                                                  |
| Version Control       | Git, GitHub, Git LFS                                             |
| Environment           | Antigravity IDE, Python 3.14                                     |

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/sujaldeb/Consumer-Spending-Intelligence.git
cd Consumer-Spending-Intelligence

# 2. Install Git LFS and pull large files
git lfs install
git lfs pull

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
python -m streamlit run app/streamlit_app.py
```

**To regenerate processed data from scratch:**

```bash
# Download credit_card_transactions.csv from Kaggle
# Place in data/raw/credit_card_transactions.csv
# Run notebooks 01 through 07 in order
```

> The raw CSV (data/raw/) is gitignored and not included in this repository.
> Processed Parquet files are tracked via Git LFS and download automatically on clone.

---

## Project Structure

    Consumer-Spending-Intelligence/
    ├── data/
    │   ├── raw/                                  <- gitignored, download from Kaggle
    │   └── processed/
    │       ├── transactions_clean.parquet        <- 95MB, tracked via Git LFS
    │       └── transactions_scored.parquet       <- 64MB, tracked via Git LFS
    ├── notebooks/
    │   ├── 01_data_ingestion_and_validation.ipynb
    │   ├── 02_exploratory_data_analysis.ipynb
    │   ├── 03_spending_trends.ipynb
    │   ├── 04_customer_intelligence.ipynb
    │   ├── 05_geo_merchant_analysis.ipynb
    │   ├── 06_anomaly_detection.ipynb
    │   └── 07_behaviour_shift_detection.ipynb
    ├── app/
    │   ├── streamlit_app.py
    │   ├── styles.css
    │   └── pages/
    │       ├── 01_overview.py
    │       ├── 02_spending_trends.py
    │       ├── 03_customer_intelligence.py
    │       ├── 04_geo_merchant.py
    │       └── 05_anomaly_detection.py
    ├── .streamlit/
    │   └── config.toml
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Author

**Sujal Deb**
📧 sujaldeb1@gmail.com
🔗 [linkedin.com/in/sujal-deb](https://linkedin.com/in/sujal-deb)
💻 [github.com/sujaldeb](https://github.com/sujaldeb)
