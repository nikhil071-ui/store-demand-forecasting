-- Star schema for analytics
CREATE TABLE dim_date (
  date_id DATE PRIMARY KEY,
  year INT,
  month INT,
  day INT,
  dayofweek INT,
  weekofyear INT,
  is_weekend INT
);

CREATE TABLE dim_store (
  store_id INT PRIMARY KEY
);

CREATE TABLE dim_item (
  item_id INT PRIMARY KEY
);

CREATE TABLE fact_sales (
  date_id DATE REFERENCES dim_date(date_id),
  store_id INT REFERENCES dim_store(store_id),
  item_id INT REFERENCES dim_item(item_id),
  sales NUMERIC,
  PRIMARY KEY (date_id, store_id, item_id)
);

-- Sample: daily sales by store, moving 7-day avg
-- (syntax may vary by warehouse)
SELECT
  s.store_id,
  d.date_id,
  SUM(f.sales) AS sales,
  AVG(SUM(f.sales)) OVER (PARTITION BY s.store_id ORDER BY d.date_id
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS sales_ma7
FROM fact_sales f
JOIN dim_store s ON s.store_id = f.store_id
JOIN dim_date d ON d.date_id = f.date_id
GROUP BY 1,2
ORDER BY 1,2;
