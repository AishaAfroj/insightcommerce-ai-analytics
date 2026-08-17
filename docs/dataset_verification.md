# Dataset verification record

Verified on 2026-08-17 before application coding.

## Official file

`e_commerce_electronic_sales_2025_dataset.csv` was downloaded from the exact
Kaggle listing named **Synthetic E-Commerce Electronic Sales 2025 Dataset**.
The listing describes a 2025 global electronics transaction simulation.

## Exact source schema

1. `order_id`
2. `date`
3. `customer_id`
4. `customer_name`
5. `customer_email`
6. `country`
7. `product`
8. `price`
9. `quantity`
10. `order_value`

## Verification outcome

| Check | Result |
|---|---:|
| Rows | 108,300 |
| Source columns | 10 |
| Missing cells | 0 |
| Duplicate rows | 0 |
| Duplicate order IDs | 0 |
| Date range | 2025-01-01 to 2025-12-31 |
| Countries | 15 |
| Products | 105 |
| Customers | 10,830 |
| Order value formula errors | 0 |
| Total simulated revenue | $133,323,271.88 |

## Schema-driven project adaptation

The file does not contain profit, discount, product category, shipping, or
cancellation fields. The implementation therefore avoids inventing those
measures. It derives only transparent calendar fields and a documented product
taxonomy from the source product names. Predictive analytics forecasts daily
revenue, while anomaly detection identifies statistically unusual transactions.

