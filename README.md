# E-Commerce & Customer Behaviour Analysis

## Project Overview

This project explores an e-commerce dataset to investigate customer behaviour, purchasing patterns, product performance, and data quality. It combines exploratory data analysis (EDA), relational analysis, statistical analysis, and RFM-based customer segmentation to derive business-relevant insights.

The project follows a structured analytical workflow, from data quality auditing and preprocessing to customer behaviour analysis and final business recommendations.

## Datasets

The project uses five related datasets:

| Dataset | Records | Description |
|---|---:|---|
| Customers | 800 | Customer information and recorded purchase metrics |
| Orders | 3,000 | Order details, payment, shipping, and delivery information |
| Order Items | 6,219 | Products purchased within individual orders |
| Products | 2,000 | Product attributes and pricing information |
| Reviews | 2,500 | Customer product reviews and ratings |

## Project Workflow

The analysis is organised into nine Jupyter notebooks.

### 1. Data Quality Audit
- Examined dataset dimensions, data types, missing values, and duplicates.
- Checked key uniqueness and relationships between datasets.
- Identified potential data quality issues.

### 2. Data Quality Investigation
- Investigated inconsistencies in customer purchase summaries.
- Examined unusual values and potential outliers.
- Assessed data integrity without making unsupported assumptions.

### 3. Univariate Exploratory Data Analysis
- Analysed individual variables and their distributions.
- Examined customer characteristics, order values, product prices, and review ratings.

### 4. Bivariate and Multivariate Exploratory Data Analysis
- Explored relationships between multiple variables.
- Investigated patterns in customer spending, order behaviour, and product performance.

### 5. Relational Analysis
- Connected customers, orders, order items, products, and reviews.
- Analysed relationships across datasets using their key identifiers.

### 6. Heavy Preprocessing
- Created processed copies of the source datasets.
- Standardised timestamp handling.
- Engineered temporal and customer-level features.
- Prepared datasets for subsequent analysis.

### 7. Statistical Analysis
- Applied correlation analysis and non-parametric statistical tests where appropriate.
- Evaluated observed relationships using statistical evidence.
- Applied multiple-testing adjustment where relevant.

### 8. Customer Behaviour Analysis
- Examined purchasing behaviour using delivered orders.
- Calculated Recency, Frequency, and Monetary (RFM) metrics.
- Assigned quartile-based RFM scores and customer segments.
- Compared segment distributions across customer tiers.

### 9. Final Findings
- Consolidated key analytical findings.
- Developed business recommendations based on the available evidence.
- Documented project limitations.

## Key Data Quality Findings

- The five source datasets contain 800 customers, 3,000 orders, 6,219 order items, 2,000 products, and 2,500 reviews.
- Referential integrity was checked across related datasets.
- Order item totals, order subtotals, and final order totals were validated.
- Recorded customer purchase metrics were compared with metrics calculated from order data.
- Recorded order counts disagreed with calculated order counts for 234 of 800 customers.
- Recorded spending totals disagreed with calculated spending totals for 234 of 800 customers.
- These discrepancies were preserved for investigation rather than silently overwritten.

## Customer Segmentation

RFM analysis was performed using delivered orders. Customers with no delivered orders were retained in the customer-level results but were not included in quartile-based scoring or the five-segment summary.

The five behavioural segments were:

- **Champions** — customers with high recency, frequency, and monetary scores.
- **Loyal Active Customers** — customers with high recency and frequency scores who do not meet all the Champions criteria.
- **Recent / Potential Customers** — customers with high recency scores but lower frequency scores.
- **At Risk - Previously Frequent** — customers with lower recency scores but high frequency scores.
- **Hibernating / Low Engagement** — customers with lower recency and frequency scores.
- **No Delivered Orders** — customers without qualifying delivered orders; retained separately and not included in the five scored segments.

The segmentation can support customer retention analysis, targeted marketing, and customer relationship management. Segment assignments depend on the selected scoring method and analysis period.

## Business Applications

The analysis can help an e-commerce business:

- Identify valuable and highly engaged customer groups.
- Investigate customers who may be at risk of disengagement.
- Understand purchasing patterns and product performance.
- Improve the reliability of customer and order reporting.
- Develop targeted customer retention and marketing strategies.

These are potential applications of the analysis rather than claims that a specific business intervention has already improved performance.

## Project Structure

```text
ecommerce-customer-behaviour/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_quality_audit.ipynb
│   ├── 02_data_quality_investigation.ipynb
│   ├── 03_univariate_eda.ipynb
│   ├── 04_bivariate_multivariate_eda.ipynb
│   ├── 05_relational_analysis.ipynb
│   ├── 06_heavy_preprocessing.ipynb
│   ├── 07_statistical_analysis.ipynb
│   ├── 08_customer_behaviour_analysis.ipynb
│   └── 09_final_findings.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── ecommerce_analysis/
│   │   ├── __init__.py
│   │   ├── customer_features.py
│   │   ├── customer_segmentation.py
│   │   ├── data_quality.py
│   │   ├── preprocessing.py
│   │   ├── relational_analysis.py
│   │   └── statistical_analysis.py
│   └── fetch_api_data.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

The `src/ecommerce_analysis/` package contains reusable analysis functions. The `notebooks/` directory contains the end-to-end analytical workflow, while `tests/` contains automated unit tests. The `data/processed/` directory contains prepared datasets used in subsequent analyses. The `reports/figures/` directory is reserved for exported figures.

## Tools and Technologies

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Relational data analysis
- Statistical hypothesis testing
- RFM customer segmentation
- Python `unittest`

## Running the Tests

Activate the project environment and navigate to the project root directory. Run:

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

The test suite covers data quality checks, preprocessing, customer features, relational calculations, statistical analysis, and RFM customer segmentation.

## Limitations

- The analysis is limited to the available datasets and their recorded information.
- Inconsistencies in customer summary metrics may affect analyses that rely on those fields.
- RFM analysis considers delivered orders and depends on the selected observation date and scoring methodology.
- Statistical relationships do not necessarily imply causation.
- Customer segments are analytical groupings and should not be interpreted as permanent customer characteristics.
- Business recommendations are based on the available data and have not necessarily been validated through real-world interventions.

## Conclusion

This project demonstrates an end-to-end analytical workflow covering data quality assessment, preprocessing, exploratory analysis, relational analysis, statistical investigation, and customer segmentation. It emphasises data integrity, reproducibility, and the use of data-driven insights to support business decision-making.