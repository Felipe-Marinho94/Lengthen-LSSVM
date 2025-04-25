# Lengthen-LSSVM
Repository containing the developed codes for obtaining the results for the paper: A New Approach for Obtain Reduced Sets in Least Squares Support Vector Machine: Lenghten via Levenberg-Marquardt

# Description
In this work a new approach for obtaining reduced sets was developed. The
heuristic consists of determining a set of initial candidates for support vectors,
which is done via vector quantization and heterogeneity assessment on the part
of the resulting clusters. Then, the pruning LM-LSSVM method is applied to the
complement of the most heterogeneous cluster, pruning the vectors in this set
with the associated Lagrange multipliers with the highest absolute values. They
are transferred to the cluster that grows with each iteration until the accuracy
evaluated in a validation set grows below a threshold. Predictive performance
and reduced processing time are proven on five benchmarking datasets. The
results indicate accuracy variations of at most 5% in relation to the LSSVM
standard for all analyzed datasets and reduction of processing time reduction in
training stage of around 99.8%.

# Examples
Examples can be found within the repository for a total of 5 different datasets:

- Haberman
- Vertebral Column (2 Class)
- Statlog German Credit
- Pima Indians Diabetes
- Breast Cancer Wisconsin
