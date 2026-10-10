# E-Commerce & Customer Behaviour Analysis

## Project Overview

This project explores an e-commerce dataset to investigate customer behaviour, purchasing patterns, product performance, and data quality. It combines exploratory data analysis (EDA), statistical analysis, relational analysis, and customer segmentation to extract meaningful business insights.

The project follows a structured workflow, from data quality auditing and preprocessing to final findings and business recommendations.

## Datasets

The project uses five related datasets:

| Dataset | Records | Description |
|---|---:|---|
| Customers | 800 | Customer information and recorded purchase metrics |
| Orders | 3,000 | Order details, payment, shipping, and delivery information |
| Order Items | 6,219 | Products purchased in individual orders |
| Products | 2,000 | Product attributes and pricing information |
| Reviews | 2,500 | Customer product reviews and ratings |

## Project Workflow

### 1. Data Quality Audit
- Examined dataset dimensions, column types, missing values, and duplicates.
- Checked primary keys and relationships between datasets.
- Identified potential data quality issues.

### 2. Data Quality Investigation
- Investigated inconsistencies in customer purchase summaries.
- Examined unusual values and potential outliers.
- Assessed data integrity without making unsupported assumptions.

### 3. Univariate EDA
- Analysed individual variables and their distributions.
- Examined customer characteristics, order values, product prices, and review ratings.

### 4. Bivariate and Multivariate EDA
- Explored relationships between multiple variables.
- Investigated patterns in customer spending, order behaviour, and product performance.

### 5. Relational Analysis
- Connected customers, orders, order items, products, and reviews.
- Analysed relationships across the datasets using their key identifiers.

### 6. Heavy Preprocessing
- Created working copies of the source datasets.
- Standardised timestamp handling.
- Engineered temporal and customer-level features.
- Saved processed datasets for subsequent analysis.

### 7. Statistical Analysis
- Applied statistical methods to investigate patterns and relationships.
- Evaluated findings using appropriate statistical evidence.

### 8. Customer Behaviour Analysis
- Examined customer purchasing behaviour.
- Developed RFM-based customer segments using delivered orders.
- Identified customer groups with different engagement and purchasing patterns.

### 9. Final Findings
- Consolidated the main analytical findings.
- Developed business recommendations based on the observed data.
- Documented project limitations.

## Key Data Quality Findings

- The five source datasets contain 800 customers, 3,000 orders, 6,219 order items, 2,000 products, and 2,500 reviews.
- Referential integrity was checked across the related datasets.
- Order item totals, order subtotals, and final order totals were validated.
- Customer-level recorded purchase metrics were compared with values calculated from order data.
- The recorded order count and spending total each disagreed with the corresponding calculated values for 234 of 800 customers.
- These discrepancies were preserved for investigation rather than silently overwritten.

## Customer Segmentation

RFM analysis was conducted using delivered orders. Customers were grouped into behavioural segments to help distinguish highly engaged customers from customers who may require additional engagement.

The identified segments included:

- Champions
- Loyal Active
- Recent/Potential
- At Risk
- Hibernating

These segments can support targeted marketing, customer retention, and customer relationship management strategies.

## Business Applications

The analysis can help an e-commerce business:

- Identify valuable and highly engaged customer groups.
- Investigate customers who may be at risk of disengagement.
- Understand purchasing patterns and product performance.
- Improve the reliability of customer and order reporting.
- Develop more targeted customer retention and marketing strategies.

## Project Structure

```text
ecommerce-customer-behaviour/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── tests/
└── README.md
```

The notebooks cover the complete analytical workflow, from data quality auditing to final findings. The `data/processed/` directory contains the prepared datasets used in subsequent analyses.

## Tools and Technologies

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Statistical analysis
- Relational data analysis
- RFM customer segmentation
- unittest

## Running the Tests

Activate the project environment, navigate to the project root directory, and run:

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

The tests validate dataset structures, key relationships, numerical calculations, customer feature consistency, and engineered order features.

## Limitations

- The analysis is limited to the available datasets and their recorded information.
- Inconsistencies in customer summary metrics may affect analyses that rely on those fields.
- Observed relationships do not necessarily imply causation.
- Customer segments depend on the selected RFM methodology and analysis period.

## Conclusion

This project demonstrates an end-to-end analytical workflow involving data quality assessment, preprocessing, exploratory analysis, statistical investigation, and customer segmentation. It emphasises data integrity, reproducibility, and the translation of analytical findings into practical business recommendations.
