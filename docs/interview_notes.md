# Vertex — Interview Discussion Notes

## Q: Walk me through the pipeline.
Vertex is a four-stage predictive analytics system: (1) Spark ingestion
reads partitioned daily JSON event payloads; (2) feature engineering
computes 30+ user-level behavioral features including temporal, engagement,
channel, and event-type ratios; (3) K-Means clustering segments users into
5 behavioral groups; (4) five classifiers are trained and compared
(NB baseline, XGBoost, LightGBM, GradientBoosting, AdaBoost).

## Q: Why 5 classifiers?
Different algorithms have different inductive biases. Naive Bayes is the
baseline (fast, interpretable). XGBoost and LightGBM are gradient boosters
optimized for tabular data. GradientBoosting is scikit-learn's implementation
for comparison. AdaBoost tests simple ensemble boosting. Comparing them on
the same features and CV splits gives a fair benchmark.

## Q: How would you scale Spark ingestion to 18GB daily?
1. Switch from local mode to YARN/K8s cluster.
2. Use Spark Structured Streaming for continuous ingestion.
3. Land events in a Delta Lake table with ACID guarantees.
4. Partition by date and hour for efficient reads.
5. The current partitioned JSON design already mirrors this pattern.

## Q: Why K-Means for segmentation?
K-Means is chosen for interpretability and speed. The cluster centroids
directly describe segment behavior (e.g., "high-value purchasers" vs.
"dormant browsers"). In production, I'd validate K with silhouette scores
and compare against DBSCAN for non-spherical clusters.

## Metrics Context
- All metrics are local-demo on 125K synthetic events.
- F1 score of ~0.84 is achievable because the synthetic data has
  clear signal between churners and non-churners.
- Real-world data would likely have lower F1 due to noise.
