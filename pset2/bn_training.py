import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pgmpy
import networkx as nx

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from pgmpy.estimators import K2
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination

from k2_greedy import k2_learn_structure
from metric_evaluations import metrics
from sklearn.metrics import confusion_matrix

# Import data from MATLAB 
mat = loadmat("data_updateBN2-1.mat")
data = mat["data"]
print(data.shape)

columns = ["SF", "F", "BD", "V", "Z", "R", "PL"]
df = pd.DataFrame(data, columns=columns)

# Randomly divide training and testing data 
train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Verify instantiations add up to 40k
print(train_df.shape)
print(test_df.shape)    

# Discretize PL
K_PL = 40

# Determine boundries from training data
quantiles = np.linspace(0, 1, K_PL + 1)
pl_edges = train_df["PL"].quantile(quantiles).to_numpy()
pl_edges = np.unique(pl_edges)   # Ensure no duplicated edges occur
print(pl_edges)

pl_edges[0] = -np.inf
pl_edges[-1] = np.inf

# Discretization for training and testing
train_df = train_df.copy()
test_df = test_df.copy()

train_df["PL_disc"] = pd.cut(
    train_df["PL"],
    bins=pl_edges,
    labels=False,
    include_lowest=True
)

test_df["PL_disc"] = pd.cut(
    test_df["PL"],
    bins=pl_edges,
    labels=False,
    include_lowest=True
)

# Add 1 to index for easier DM/DEM interpratition
train_df["PL_disc"] += 1
test_df["PL_disc"] += 1

print(train_df["PL_disc"].value_counts().sort_index())

# Discretize input variables
def state_mapping(series):
    values = np.sort(series.unique())

    return{
        value: state
        for state, value in enumerate(values, start=1) 
    }

input_cols = ["SF", "F", "BD", "V", "Z", "R"]
state_mappings = {}

# Create the mappings for training data
for col in input_cols:
    state_mappings[col] = state_mapping(train_df[col])

# Use mappings
for col in input_cols:
    train_df[col + "_disc"] = (
        train_df[col].map(state_mappings[col])
    )

    test_df[col + "_disc"] = (
        test_df[col].map(state_mappings[col])
    )

# Check for missing states
for col in input_cols:
    name = col + "_disc"
    print(name, test_df[name].isna().sum())

# Build discretized dataset
disc_columns = [
    "SF_disc",
    "F_disc",
    "BD_disc",
    "V_disc",
    "Z_disc",
    "R_disc",
    "PL_disc",
]

train_disc = (train_df[disc_columns].copy())
test_disc = (test_df[disc_columns].copy())

## K2 Greedy Structure Learning
# Create initial node  guess here
node_order = [
    "SF_disc",
    "F_disc",
    "BD_disc",
    "V_disc",
    "Z_disc",
    "R_disc",
    "PL_disc"
]

max_parents = 3     # Limit parents so CPT is small

## K2 greedy search structure function
parents = k2_learn_structure(train_disc, node_order, max_parents)

print("\nLearned search structure:")
for node, node_parents in parents.items():
    print(
        f"{node}: {node_parents}"
    )

# Implement edges from learned parents
edges = []

for child, child_parents in parents.items():
    for parent in child_parents:
        edges.append((parent, child))

print("\nK2 learned edges:")
for edge in edges:
    print(edge)


## Create Bayesian Network
model = DiscreteBayesianNetwork(edges)
model.add_nodes_from(node_order)    # Explicility add every variable even if no edges
print("\nBayesian network:")
print(model.nodes())
print(model.edges())

## Expectation-Maximization Algorithim 
em = ExpectationMaximization(model=model, data=train_disc)
cpts = em.get_parameters(max_iter=100, atol=1e-8, n_jobs=1, show_progress=True)
model.add_cpds(*cpts)

print("Model valid", model.check_model())

# Print learned CPTs
for cpd in model.get_cpds():
    print(cpd)
    print()

## Create inference
inference = VariableElimination(model)

row = test_disc.iloc[0]

print("\nFirst test sample:")
print(row)

evidence = {
    "SF_disc": int(row["SF_disc"]),
    "F_disc": int(row["F_disc"]),
    "BD_disc": int(row["BD_disc"]),
    "V_disc": int(row["V_disc"]),
    "Z_disc": int(row["Z_disc"]),
    "R_disc": int(row["R_disc"])
}

actual_state = int(row["PL_disc"])

posterior = inference.query(
    variables=["PL_disc"],
    evidence=evidence,
    show_progress=False
)

print("\nPosterior:")
print(posterior)

print("\nProbabilities:")
print(posterior.values)

result = metrics(posterior, actual_state)

print("\nSingle-sample metrics:")
print(result)

example_posterior = posterior
example_actual_state = actual_state
## Evaluate all test observations
results = []
for _, row in test_disc.iterrows():

    evidence = {
        "SF_disc": int(row["SF_disc"]),
        "F_disc": int(row["F_disc"]),
        "BD_disc": int(row["BD_disc"]),
        "V_disc": int(row["V_disc"]),
        "Z_disc": int(row["Z_disc"]),
        "R_disc": int(row["R_disc"])
    }

    actual_state = int(row["PL_disc"])

    posterior = inference.query(
        variables=["PL_disc"],
        evidence=evidence,
        show_progress=False
    )

    result = metrics(posterior, actual_state)
    results.append(result)

results_df = pd.DataFrame(results)
print(results_df.head())

MA = results_df["MA"].mean()
DM = results_df["DM"].mean()
DEM = results_df["DEM"].mean()
NDDE = results_df["NDDE"].mean()

print("\nBN Test Results")
print("----------------------")
print(f"MA   = {MA:.4f}")
print(f"DM   = {DM:.4f}")
print(f"DEM  = {DEM:.4f}")
print(f"NDDE = {NDDE:.4f}")

cm = confusion_matrix(
    results_df["Actual"],
    results_df["Predicted"]
)

print("\nConfusion Matrix:")
print(cm)

labels = np.arange(1, cm.shape[0] + 1)

plt.figure(figsize=(10, 9))
plt.imshow(cm)
plt.colorbar()
plt.xticks(np.arange(len(labels)), labels, rotation=90)
plt.yticks(np.arange(len(labels)), labels)
plt.xlabel("Predicted PL State")
plt.ylabel("Actual PL State")
plt.title("PL State Confusion Matrix")
plt.show()


G = nx.DiGraph()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())

pos = {
    "SF_disc": (0, 3),
    "F_disc":  (0, 2),
    "BD_disc": (0, 1),
    "V_disc":  (2, 1),
    "Z_disc":  (0, 0),
    "R_disc":  (0, -1),
    "PL_disc": (3, 0)
}

node_labels = {
    "SF_disc": "SF",
    "F_disc": "F",
    "BD_disc": "BD",
    "V_disc": "V",
    "Z_disc": "Z",
    "R_disc": "R",
    "PL_disc": "PL"
}

plt.figure(figsize=(9, 6))
nx.draw_networkx(
    G,
    pos=pos,
    labels=node_labels,
    with_labels=True,
    node_size=2200,
    font_size=11,
    arrows=True,
    arrowsize=20
)
plt.title("Learned Bayesian Network Structure")
plt.axis("off")
plt.tight_layout()
plt.show()


finite_edges = pl_edges[np.isfinite(pl_edges)]

plt.figure(figsize=(9, 5))
plt.hist(
    train_df["PL"],
    bins=50
)

for edge in finite_edges:
    plt.axvline(
        edge,
        linestyle="--",
        linewidth=1
    )

plt.xlabel("Propagation Loss, PL")
plt.ylabel("Training Sample Count")
plt.title(f"PL Distribution and Quantile Boundaries, K_PL = {K_PL}")
plt.tight_layout()
plt.show()


pl_counts = train_disc["PL_disc"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(
    pl_counts.index,
    pl_counts.values
)
plt.xticks(np.arange(1, K_PL + 1))
plt.xlabel("PL Discrete State")
plt.ylabel("Training Sample Count")
plt.title(f"PL State Counts After Quantile Discretization, K_PL = {K_PL}")
plt.tight_layout()
plt.show()


cm_norm = confusion_matrix(
    results_df["Actual"],
    results_df["Predicted"],
    normalize="true"
)

labels = np.arange(1, cm_norm.shape[0] + 1)

plt.figure(figsize=(10, 9))
plt.imshow(
    cm_norm,
    vmin=0,
    vmax=1
)
plt.colorbar()
plt.xticks(np.arange(len(labels)), labels, rotation=90)
plt.yticks(np.arange(len(labels)), labels)
plt.xlabel("Predicted PL State")
plt.ylabel("Actual PL State")
plt.title("Normalized PL State Confusion Matrix")
plt.tight_layout()
plt.show()


posterior_probabilities = example_posterior.values

posterior_states = np.asarray(
    example_posterior.state_names["PL_disc"],
    dtype=int
)

predicted_state = posterior_states[
    np.argmax(posterior_probabilities)
]

plt.figure(figsize=(8, 5))
plt.bar(
    posterior_states,
    posterior_probabilities
)
plt.axvline(
    example_actual_state,
    linestyle="--",
    linewidth=2,
    label=f"Actual = {example_actual_state}"
)
plt.axvline(
    predicted_state,
    linestyle=":",
    linewidth=2,
    label=f"MAP = {predicted_state}"
)
plt.xticks(posterior_states)
plt.xlabel("PL State")
plt.ylabel("Posterior Probability")
plt.title("Posterior Distribution for Example Test Observation")
plt.legend()
plt.tight_layout()
plt.show()

dm_counts = results_df["DM"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(
    dm_counts.index,
    dm_counts.values
)
plt.xticks(dm_counts.index)
plt.xlabel("Distance Metric")
plt.ylabel("Number of Test Samples")
plt.title("Distribution of PL Prediction Distance")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 7))
plt.scatter(
    results_df["Actual"],
    results_df["Predicted"],
    alpha=0.15,
    s=10
)

min_state = int(
    min(
        results_df["Actual"].min(),
        results_df["Predicted"].min()
    )
)

max_state = int(
    max(
        results_df["Actual"].max(),
        results_df["Predicted"].max()
    )
)

plt.plot(
    [min_state, max_state],
    [min_state, max_state],
    linestyle="--"
)
plt.xticks(np.arange(min_state, max_state + 1))
plt.yticks(np.arange(min_state, max_state + 1))
plt.xlabel("Actual PL State")
plt.ylabel("Predicted PL State")
plt.title("Actual vs Predicted PL State")
plt.grid(True)
plt.tight_layout()
plt.show()