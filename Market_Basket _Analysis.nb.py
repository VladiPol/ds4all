# Databricks notebook source
# MAGIC %md
# MAGIC # Shopping basket analysis of a bakery
# MAGIC #### Databricks Free Edition Hackathon from November 5-14, 2025
# MAGIC author: "Vladimir Poliakov"

# COMMAND ----------

# MAGIC %md
# MAGIC ##1 Introduction
# MAGIC Every single company, that sells something, automatically has the data, whether in electronic format or on paper, for shopping or market basket analysis. And bakeries are no exception in this regard. The large retailers use usual the Apriori algorithm. This notebook should present this algorithm on the sales transactions dataset one small bakery. The scope is to demonstrate here, even the small bakery can get more profit from shopping basket analysis or in other words from Data Science.  

# COMMAND ----------

# MAGIC %md
# MAGIC ##2 Theory
# MAGIC As menshioned above for the market will be used basket the Apriori algorithm. The algorithm tries to uncover associations between items in the basket (the searching for the strong rules also known as Association Rules).
# MAGIC
# MAGIC Please check in the following links for more information:
# MAGIC * Association rule learning on Wikipedia: https://en.wikipedia.org/wiki/Association_rule_learning
# MAGIC * KDnuggets: https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html
# MAGIC
# MAGIC The key metrics the Apriori algorithm are: Support, Confidence and Lift.
# MAGIC
# MAGIC **Support**
# MAGIC Support answers the question, how popular an itemset is, or how often appear an itemset in all transactions.<br>
# MAGIC support(A,B) = (transactions with A and B) / (total Transaktionen)
# MAGIC
# MAGIC **Confidence**
# MAGIC The confidence (A -> A) indicates how often A is purchased when B is purchased.<br>
# MAGIC confidence(A→B) = (transactions with A and B) / (transactions with A)
# MAGIC
# MAGIC **Lift**
# MAGIC The lift provides the answer to the question of how much more likely A makes the purchase of B.<br>
# MAGIC lift(A,B) = support(A,B) / (support(A) x support(B))

# COMMAND ----------

# MAGIC %md
# MAGIC ##3 Loading data into Bronze table
# MAGIC The dataset was downloaded from Kaggle service https://www.kaggle.com/sulmansarwar/transactions-from-a-bakery
# MAGIC
# MAGIC Dataset description from Kaggle:
# MAGIC
# MAGIC _The dataset consists of 21293 observations from a bakery. The data file contains four variables, Date, Time, Transaction ID and Item. Transaction ID ranges from 1 through 9684. However, there are some skipped numbers in Transaction IDs. Also, there are duplicated entries, as shown in observation # 2 and #3. Besides, the Item contains "Adjustment", "NONE", and "Afternoon with the baker". While the entries of "Adjustment" and "NONE" are straight forward, "Afternoon with the baker" may be a real purchase._
# MAGIC
# MAGIC The statemet about duplicated entries is not really correct. It could be really many items in the shopping basket, what means the item was bought double, three tiems etc.

# COMMAND ----------

from pyspark.sql import functions as F

# It is not really neccessary to transform the CSV into bronze table in this case, but we do following best practices
# Read the CSV an transform into bronze table: Raw Ingestion
bronze_df = spark.read.csv("/Volumes/workspace/default/input/BreadBasket_DMS.csv", header=True, inferSchema=True, sep=",")
bronze_df = bronze_df.withColumn("Ingest_ts", F.current_timestamp())
bronze_df.write.mode("overwrite").saveAsTable("workspace.default.bronze_basket_transactions")

bronze_df.printSchema()
display(bronze_df)

# COMMAND ----------

# Let's have a look at the statistics
bronze_df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##4 Cleaning data and loading it into Silver table

# COMMAND ----------

# As menshoned above delete all transaction with "Adjustment", "NONE", and "Afternoon with the baker",
# because we want to know the real existing items
silver_df = bronze_df.filter(bronze_df.Item.isNotNull() & bronze_df.Transaction.isNotNull())

# Filer only items, which do NOT have "Adjustment", "NONE", and "Afternoon with the baker" values
silver_df = silver_df.filter(~(bronze_df.Item.isin(["Adjustment", "NONE", "Afternoon with the baker"])))

silver_df = silver_df \
    .withColumn("Year", F.year("Date")) \
    .withColumn("Month", F.month("Date")) \
    .withColumn("Day", F.dayofmonth("Date")) \
    .withColumn("Hour", F.hour("Time")) \
    .withColumn("Minute", F.minute("Time"))

silver_df.write.mode("overwrite").saveAsTable("workspace.default.silver_basket_transactions")
silver_df.printSchema()
display(silver_df)

# COMMAND ----------

# Finally we have 831 rows less
silver_df.describe().show()

# COMMAND ----------

# Let's have a look at the unique items. We should have 92 unique items
unique_items = silver_df.select("Item").distinct()
assert unique_items.count() == 92, "There should be 92 unique items"
display(unique_items)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 Visualizing and Understanding the Data
# MAGIC The dataset has the transactions from 2016-10-30 until 2017-04-09. Let's try to answer for some questions
# MAGIC * Which items do customers buy most? 
# MAGIC * Which months were more successful?
# MAGIC * Which hours are the rush hours?

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Which items do customers buy most? Let see the 10 most popular items
# MAGIC -- Logically the most popular items should be Coffee and Bread
# MAGIC SELECT Item, count(Item) AS Count_Item FROM default.silver_basket_transactions GROUP BY Item ORDER BY Count_Item DESC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Which months were more successful?
# MAGIC -- The most successful month was November, but there are not data for some months
# MAGIC SELECT Month, count(Item) AS Count_Item FROM default.silver_basket_transactions GROUP BY ALL ORDER BY Count_Item DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Which hours are the rush hours?
# MAGIC SELECT Hour, count(Item) AS Count_Item FROM default.silver_basket_transactions GROUP BY ALL ORDER BY Hour, Count_Item;

# COMMAND ----------

# MAGIC %md
# MAGIC ##6 Shopping basket analysis
# MAGIC For the small to medium datasets we can use `mlxtend` package, because it’s simpler and Pandas-based. However, approach in a Databricks environment is to leverage its distributed computing power from the start. For that reason, using the `FPGrowth` class from Spark MLlib is more preferable, since the order of items in the basket is not important. `FPGrowth` is designed for distributed processing and works perfectly in Databricks, making it future-proof if the dataset grows significantly.
# MAGIC
# MAGIC **Warning**: Since the Databricks Free Editon is limited to Serverless Compute and the solution with `FPGrowth` does not work _[CONNECT_ML.UNSUPPORTED_EXCEPTION] Generic Spark Connect ML error. FPGrowth algorithm is not supported if Spark Connect model cache offloading is enabled. SQLSTATE: XX000_, let us try to implement Market Basket Analysis using pure PySpark SQL.
# MAGIC
# MAGIC For the market basket analysis using FP-growth on Databricks s. the blog article: https://www.databricks.com/blog/2018/09/18/simplify-market-basket-analysis-using-fp-growth-on-databricks.html

# COMMAND ----------

# MAGIC %sql
# MAGIC -----------------------------
# MAGIC -- Pure PySpark SQL Solution
# MAGIC -- ¯\_(ツ)_/¯
# MAGIC -----------------------------
# MAGIC -- Generate Item Pairs per Transaction (excluding self-pairs)
# MAGIC CREATE OR REPLACE TEMP VIEW item_pairs AS
# MAGIC SELECT
# MAGIC     a.Item AS item_A,
# MAGIC     b.Item AS item_B,
# MAGIC     a.Transaction
# MAGIC FROM default.silver_basket_transactions a
# MAGIC JOIN default.silver_basket_transactions b
# MAGIC ON a.Transaction = b.Transaction
# MAGIC WHERE a.item != b.item;  -- avoid duplicates and self-pairs
# MAGIC
# MAGIC -- Total number of transactions
# MAGIC CREATE OR REPLACE TEMP VIEW total_tx AS
# MAGIC SELECT COUNT(DISTINCT Transaction) AS total_tx
# MAGIC FROM default.silver_basket_transactions;
# MAGIC
# MAGIC -- Item frequency
# MAGIC CREATE OR REPLACE TEMP VIEW item_freq AS
# MAGIC SELECT item, COUNT(DISTINCT Transaction) AS item_count
# MAGIC FROM default.silver_basket_transactions
# MAGIC GROUP BY item;
# MAGIC
# MAGIC -- Pair frequency
# MAGIC CREATE OR REPLACE TEMP VIEW pair_freq AS
# MAGIC SELECT item_A, item_B, COUNT(DISTINCT Transaction) AS pair_count
# MAGIC FROM item_pairs
# MAGIC GROUP BY item_A, item_B;
# MAGIC
# MAGIC -- Calculate Support, Confidence, Lift and safe it into gold table for late analysis by Genie
# MAGIC CREATE OR REPLACE TABLE gold_basket_transactions AS
# MAGIC SELECT
# MAGIC     pf.item_A,
# MAGIC     pf.item_B,
# MAGIC     pf.pair_count,
# MAGIC     iA.item_count AS item_A_count,
# MAGIC     iB.item_count AS item_B_count,
# MAGIC     ROUND(pf.pair_count / t.total_tx, 4) AS support,
# MAGIC     ROUND(pf.pair_count / iA.item_count, 4) AS confidence_A_to_B,
# MAGIC     ROUND(pf.pair_count / iB.item_count, 4) AS confidence_B_to_A,
# MAGIC     ROUND((pf.pair_count / t.total_tx) / ((iA.item_count / t.total_tx) * (iB.item_count / t.total_tx)), 4) AS lift
# MAGIC FROM pair_freq pf
# MAGIC JOIN item_freq iA ON pf.item_A = iA.item
# MAGIC JOIN item_freq iB ON pf.item_B = iB.item
# MAGIC CROSS JOIN total_tx t
# MAGIC ORDER BY lift DESC;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##7 Conclusion
# MAGIC
# MAGIC * Using SQL instead of the FPGrowth class from Spark MLlib for Market Basket Analysis may not be the best solution, especially when dealing with many products in the market basket. However, it is sufficient for a hackathon to demonstrate an alternative Databricks distributed solution.
# MAGIC * The Genie in the Databricks platform helps perform analysis in descriptive language without requiring knowledge of SQL.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC * Dataset from Kaggle:<br>
# MAGIC https://www.kaggle.com/datasets/sulmansarwar/transactions-from-a-bakery
# MAGIC * Affinity analysis on Wikipedia:<br>
# MAGIC https://en.wikipedia.org/wiki/Affinity_analysis
# MAGIC * Association Rules and the Apriori Algorithm: A Tutorial on KDnuggets:<br>
# MAGIC https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html
# MAGIC * Simplify Market Basket Analysis using FP-growth on Databricks:<br>
# MAGIC https://www.databricks.com/blog/2018/09/18/simplify-market-basket-analysis-using-fp-growth-on-databricks.html
# MAGIC * Warenkorbanalyse einer Bäckerei (Repo on GitHub in German):<br>
# MAGIC https://github.com/VladiPol/ds4all/blob/master/baeckerei_warenkorbanalyse.ipynb
# MAGIC

# COMMAND ----------

