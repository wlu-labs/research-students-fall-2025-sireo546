# ============================================================
# # @file  compare script.py
# Quantization Comparison: PTQ, QAT, GPTQ
# ============================================================
# Compare how three quantization methods affect performance.
# Measure accuracy, speed, and model size.
# @author Rameen Amin
# @version 2025-10-14
# Dataset: Traffic congestion data 
# ============================================================

#imports

import torch                      # PyTorch: used to create and train models
import torch.nn as nn             # For building neural networks
import torch.optim as optim       # For optimization (training)
import time                       # To measure how long things take
import os                         # To check file sizes (model size)
import tempfile
import pandas as pd               # To load and handle CSV data
import numpy as np                # For working with numbers and arrays
torch.manual_seed(42)
np.random.seed(42)
from sklearn.model_selection import train_test_split   # To split data into training/testing sets
from sklearn.preprocessing import StandardScaler        # To scale input values (normalization)
from sklearn.metrics import accuracy_score              # To calculate model accuracy
from torch.ao.quantization import (                     # Tools for quantization
    quantize_dynamic, 
    get_default_qconfig, 
    get_default_qat_qconfig,
    prepare, 
    prepare_qat, 
    convert
)

#measure accuracy properly
def get_model_size(model):
    """
    Saves model temporarily and returns exact size in MB(high precision). More accurate than using os.pathgetsize() directly.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(),tmp.name)
        size_mb = os.path.getsize(tmp.name)/(1024 *1024) #bytes to MB
    os.remove(tmp.name)
    return round(size_mb, 6)


import platform 

# ============================================================
# Automatically select quantization backend 
# ============================================================
is_arm = platform.machine().lower() in ("arm64", "aarch64")  # Apple Silicon is arm64
if is_arm:
    torch.backends.quantized.engine = "qnnpack"
    print("Quantization engine set to qnnpack (Apple Silicon / ARM64).")
else:  # x86_64 (Beluga, Narval, etc.)
    torch.backends.quantized.engine = "fbgemm"
    print("Quantization engine set to fbgemm (x86 CPU).")

# Detect if running on macOS (used later to skip unsupported quantized ops)
is_mac = platform.system() == "Darwin"


#  STEP 1: LOAD AND PREPARE  TRAFFIC DATASET
# Load  traffic data
df = pd.read_csv("trafficdata.csv")

# Pick features 
features = [
    "total_vehicle", "am_peak_vehicle", "pm_peak_vehicle",
    "n_appr_vehicle", "e_appr_vehicle", "s_appr_vehicle", "w_appr_vehicle"
]

# Drop rows with missing values in these columns
# dropna() removes incomplete rows.
df = df.dropna(subset=features)

# classifing traffic is easier than prediciting numbers so we'll do that

# Ccreate 3 congestion levels (Low, Medium, High) based on total vehicle count
p33, p67 = np.percentile(df["total_vehicle"], [33, 67])  # calculate 33rd and 67th percentile
def label_congestion(v):
    if v < p33: #If total vehicles < 33rd percentile, low congestion
        return 0
    elif v < p67: # If between 33rd–67th percentile, medium congestion
        return 1
    else: # Else , high congestion
        return 2

# Apply same function to “total_vehicle” column for the new labels
df["label"] = df["total_vehicle"].apply(label_congestion)

    
    # Apply same function to “total_vehicle” column for the new labels
df["label"] = df["total_vehicle"].apply(label_congestion)

# X = input features ( model will learn from this data)
# y = output labels (model will try to predict answers)
X = df[features].values.astype(np.float32)
y = df["label"].values.astype(np.int64)


# Split the data: 80% for training, 20% for testing
# The training part is used to teach the model
# The testing part checks how well it learned
# random_state ensures the same random split every time (for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale (normalize) the data so that values are centered around 0
# This helps training be faster and prevents large numbers from dominating
# Normalization does help to prevent features with large values (like 10,000 vehicles)
# from overpowering smaller ones (like 50 bikes). It rescales everything to
# roughly the same range (mean 0, std 1)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch 
# PyTorch models only work with tensor data types.
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# 3️ DEFINE A SIMPLE BASELINE MODEL

# This section builds a basic neural network that will predict congestion levels
# (low / medium / high) from the traffic data.
# This model is the "original" FP32 (full precision) version —
# later it will be quantized to compare the three methods (PTQ, QAT, GPTQ).


# Every PyTorch model must inherit from nn.Module

class TrafficNet(nn.Module):
    def __init__(self):
        # super() initializes the parent class (nn.Module)
        super().__init__()

        # First layer (fully connected or "dense" layer)
        # Input = 7 numbers (the features: total_vehicle, am_peak_vehicle, etc.)
        # Output = 64 neurons (internal hidden layer that learns patterns)
        self.fc1 = nn.Linear(7, 64)

        # ReLU activation adds non-linearity, helping the network
        # learn complex relationships instead of just straight lines.
        self.relu = nn.ReLU()

        # Second layer: reduces the 64 hidden neurons into 3 outputs.
        # Why 3? Because we have 3 congestion levels (low, medium, high).
        self.fc2 = nn.Linear(64, 3)

    # Defines how the data moves through the model (the forward pass)
    def forward(self, x):
        # Step 1: pass input through first layer (7 → 64)
        # Step 2: apply ReLU activation to make outputs non-linear
        # Step 3: pass through final layer (64 → 3)
        return self.fc2(self.relu(self.fc1(x)))



# Create the model

# Here we instantiate (create an instance of) the TrafficNet model.
# It starts with random weights that will be updated during training.
base = TrafficNet().to(device)



# Choose optimization and loss function
#Understand this, what is the loss function??

# The optimizer decides how weights are updated to reduce error.
# "Adam" automatically adapts the learning rate for each parameter.
opt = optim.Adam(base.parameters(), lr=0.001)

# The loss function measures how wrong the model’s predictions are.
# CrossEntropyLoss is used for classification problems (multiple classes).
loss_fn = nn.CrossEntropyLoss()


# TRAIN THE BASELINE (FP32) MODEL

# Purpose:
# To teach the model to understand the relationship between the
# vehicle counts and congestion labels (0, 1, 2).
# Each "epoch" is one full pass through all training examples.

for epoch in range(15):  # 15 rounds of training
    # Forward pass: the model makes predictions for all training samples.
    out = base(X_train)

    # Compute how far predictions are from the actual labels.
    # Lower loss = better performance.
    loss = loss_fn(out, y_train)

    # Reset gradients before every backpropagation step
    # (otherwise gradients would add up from previous loops)
    opt.zero_grad()

    # Backpropagation:
    # Calculates how each parameter contributed to the error.
    loss.backward()

    # Update model weights using calculated gradients.
    opt.step()

    # Print progress for every epoch to see if loss decreases over time.
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")


# Evaluate baseline model on test data

# After training, test how well the model performs on unseen data.
# This gives  the accuracy before any quantization.

# Disable gradient calculations (saves time and memory during evaluation)
with torch.no_grad():
    # The model outputs raw scores for each class.
    # argmax(dim=1) picks the class (0, 1, or 2) with the highest score.
    pred = base(X_test).argmax(dim=1)

# accuracy_score compares predictions vs real answers.
baseline_acc = accuracy_score(y_test, pred)



# Save model and check file size
#
# Save the trained model’s weights to a file for future use.
torch.save(base.state_dict(), "base_fp32.pt")

# Check the size of the saved model (in MB).
base_size = get_model_size(base)

print(f"\nFP32 Accuracy: {baseline_acc*100:.2f}%  | Size: {base_size:.2f} MB")



# measure speed with this helper function 
# Quantization also improves speed, so measure average
# time (latency) each model takes per prediction.
# Each latency value below is the average time (in milliseconds)
# it takes for the model to process one batch of test data.
# Averaged across 100 runs for consistency.

def measure_latency(model, data, device = "cpu", runs = 100):
    model.eval()
    data = data.to(device)


    # Turn off gradient calculations
    with torch.no_grad():
        # Run the model once on the dataset
        _ = model(data)
    
     # Record start time
    start_time = time.perf_counter()
    for _ in range(runs):
        _ = model(data)
    end_time = time.perf_counter()


    total_time = (end_time - start_time) * 1000 #convert to miliseconds
    avg_latency = total_time / runs
    return round(avg_latency, 4) # 4 decimal placs 


# ------------------------------------------------------------
#  POST TRAINING QUANTIZATION (PTQ)
# ------------------------------------------------------------
# Purpose:
# Simplest and most commonly used quantization method.
# Instead of training a new model, we take the already-trained FP32 model
# and convert its weights into int8 (8-bit integers) after training.
#
# Why?
# - It makes  model smaller (uses less storage and memory)
# - It runs faster (because 8-bit math is faster than 32-bit)
# - quick — no retraining needed
#
# Downside:
# - rounding and compression happen after training, therefore
#   it can slightly reduce accuracy (the model wasn’t trained
#   to handle the precision loss).
# ------------------------------------------------------------


# ------------------------------------------------------------
# quantize_dynamic(): does PTQ automatically
# ------------------------------------------------------------
# quantize_dynamic() is a built-in PyTorch function that automatically converts certain layers (like nn.Linear) to int8 format.
#
# Doesn’t change the model’s structure, only the precision of its weights.
# Have to tell it:
# - which model to quantize (base)
# - which layers to quantize ({nn.Linear})
# - and which data type to use (torch.qint8 means quantized int8)
# Ensure model is on CPU before quantization
base_cpu = base.to("cpu")
torch.backends.quantized.engine = "qnnpack"  # Apple Silicon CPU test

ptq_model = quantize_dynamic(base_cpu, {nn.Linear}, dtype=torch.qint8)# Save the quantized model to a file

# Saving the quantized model allows you to check how much

# Put PTQ model in eval mode for inference
ptq_model.eval()

# Save the PTQ model’s weights
torch.save(ptq_model.state_dict(), "ptq_model.pt")

# Size on disk
ptq_size = get_model_size(ptq_model)

# Accuracy on test data
with torch.no_grad():
    pred_ptq = ptq_model(X_test).argmax(dim=1)
ptq_acc = accuracy_score(y_test, pred_ptq)

# Latency (ms/sample)
ptq_latency = measure_latency(ptq_model, X_test)

print(f"\nPTQ  → Acc: {ptq_acc*100:.2f}% | Latency: {ptq_latency:.2f} ms | Size: {ptq_size:.2f} MB")




# ------------------------------------------------------------
#  QUANTIZATION AWARE TRAINING (QAT)
# ------------------------------------------------------------
# Purpose:
# QAT more advanced form of quantization.
# Instead of converting a finished model, QAT pretends during training
# that parts of the model are already quantized (8-bit) — this is called
# “fake quantization.”
#
# Why?
# - It lets the model adapt to quantization effects while still learning.
# - The final quantized model keeps accuracy closer to the original FP32.
# - It’s slower than PTQ because it involves retraining.
#
# ------------------------------------------------------------


# Create a fresh copy of the baseline model

# start with a new version of the same architecture (TrafficNet)
# so that  don’t modify our already-trained PTQ model.
qat_model = TrafficNet()

# Attach quantization configuration (qconfig)

# PyTorch needs to know how to simulate quantization.
# The “qconfig” defines:which quantization backend to use (e.g., "fbgemm" for CPU)
#how activations and weights will be quantized (per-tensor or per-channel)
#
# get_default_qat_qconfig() gives us a standard, reliable setup for int8.
qat_model.qconfig = get_default_qat_qconfig("fbgemm")



# Prepare the model for QAT

# prepare_qat() inserts fake quantization layers into the model.
# The layers:
#   - round values as if they were quantized to int8
#   - still allow gradients to pass in FP32
# This lets the network “practice” being quantized during training.
qat_prep = prepare_qat(qat_model)



# Fine-tune (train) the QAT model

# To train again a little so the model can adjust its weights
# to reduce the accuracy drop caused by quantization rounding.
#
#don’t need long training, just a few epochs to let it adapt.
opt_q = optim.Adam(qat_prep.parameters(), lr=0.001)

for epoch in range(3):  # small retraining loop
    # Forward pass: make predictions on training data
    out = qat_prep(X_train)

    # Calculate loss (how far predictions are from true labels)
    loss = loss_fn(out, y_train)

    # Clear previous gradients
    opt_q.zero_grad()

    # Backpropagate (find how weights should change)
    loss.backward()

    # Apply the weight updates
    opt_q.step()

    # Print progress to track learning
    print(f"QAT epoch {epoch+1}: loss = {loss.item():.4f}")



# Convert fake quantized model to real int8 model

# Once training finishes, i replace the fake quantization layers
# with real quantization operations so it becomes an actual int8 model.
qat_model.to("cpu")
qat_prep.to("cpu")
qat_int8 = convert(qat_prep.eval()) # eval() sets model to evaluation mode

# Save and check model size

# Save the quantized weights to a file for size comparison.
torch.save(qat_int8.state_dict(), "qat_model.pt")

# Get model size in MB to compare with FP32 and PTQ.
qat_size = get_model_size(qat_int8)



# Look at QAT model accuracy
# Check how accurate the quantized version is on test data
with torch.no_grad():
    # The converted int8 model can't run on macOS CPU backend, so instead
# evaluate the model *before* conversion (still simulated quantization).
    pred_qat = qat_model(X_test).argmax(dim=1)
  # pick most likely class
qat_acc = accuracy_score(y_test, pred_qat)      # compute % of correct predictions




if is_mac:
    print("\n[Note] Real int8 execution isn’t supported on macOS (qnnpack CPU), so latency is measured using the pre-conversion model to simulate quantized performance.")
    qat_latency = measure_latency(qat_prep, X_test)
else:
    qat_latency = measure_latency(qat_int8, X_test)
# Measure latency (speed)
# use the same helper function as before to check how long each prediction takes on average.
#qat_latency = measure_latency(qat_int8, X_test)

# On macOS (ARM64), quantized::linear can't run after conversion.
# measure latency on the fake-quantized model instead to avoid backend crash.
try:
    qat_latency = measure_latency(qat_int8, X_test)
except NotImplementedError:
    print("\n[Note] Real int8 execution isn’t supported on macOS (qnnpack CPU), so latency is measured using the pre-conversion model to simulate quantized performance")
    qat_latency = measure_latency(qat_prep, X_test)


# Print results for QAT

print(f"\nQAT → Acc: {qat_acc*100:.2f}% | Latency: {qat_latency:.2f} ms | Size: {qat_size:.2f} MB")

# ------------------------------------------------------------
#  GPTQ (Gradient Post-Training Quantization)
# ------------------------------------------------------------

# GPTQ (Gradient Post-Training Quantization) is a smarter version of PTQ.
# Instead of just rounding weights to 8-bit (like PTQ),
# GPTQ makes a *small correction* using gradient information
# to make sure the quantized weights behave more like the original FP32 weights.
#
# In other words:
# - PTQ: just compresses weights, may lose accuracy.
# - GPTQ: compresses + adjusts weights slightly using math (gradients)
#         → keeps higher accuracy even at low bit-widths (like 4-bit).
#
#  used for quantizing big LLMs to fit on local devices.
# ------------------------------------------------------------


# ------------------------------------------------------------
# In this simplified experiment,  simulate GPTQ behavior.
# ------------------------------------------------------------
# Real GPTQ needs a large GPU and external libraries like "AutoGPTQ" or "bitsandbytes".
# Simulate what GPTQ does conceptually
# — the results still show the trade-offs.
#
# Later, can install AutoGPTQ and run the true version.

#  take the FP32 baseline accuracy as our reference point.
# GPTQ usually causes only a small accuracy drop (like 0.5–1%).
gptq_acc = baseline_acc - 0.01           # assume around 1% drop in accuracy

# GPTQ typically gives very compact models (4-bit weights = 1/8 the size of FP32)
# But some overhead exists, so roughly 50% smaller than FP32 is realistic.
gptq_size = base_size * 0.5            # half the FP32 model size

# GPTQ is also faster because lower bit precision = fewer computations.
# have to assume it runs about 20% faster than PTQ on average.
gptq_latency = ptq_latency * 0.8



# Print GPTQ comparison results

print(f"\nGPTQ → Acc: {gptq_acc*100:.2f}% | Latency: {gptq_latency:.2f} ms | Size: {gptq_size:.2f} MB")
print("Here, GPTQ latency and size are simulated estimates — actual gradient-based quantization will be tested on Béluga")
# ------------------------------------------------------------
# 8️ FINAL COMPARISON SUMMARY
# ------------------------------------------------------------

# We now combine results from all four models:
#   1. FP32 (original full precision model)
#   2. PTQ  (post-training quantization)
#   3. QAT  (quantization-aware training)
#   4. GPTQ (gradient post-training quantization)
#
#  summarize them in a single table so it’s easy to compare
# accuracy, size, and latency side-by-side.
# ------------------------------------------------------------


# ------------------------------------------------------------
# Combine all results into a single table
# ------------------------------------------------------------
#  create a pandas DataFrame that lists:
# - Method name
# - Accuracy percentage (%)
# - Model file size (MB)
# - Latency per sample (ms)
results = pd.DataFrame([
    ["FP32 (Original)", baseline_acc*100, base_size, measure_latency(base, X_test)],
    ["PTQ (Post-Training)", ptq_acc*100, ptq_size, ptq_latency],
    ["QAT (Aware Training)", qat_acc*100, qat_size, qat_latency],
    ["GPTQ (Gradient PTQ)", gptq_acc*100, gptq_size, gptq_latency]
], columns=["Method", "Accuracy (%)", "Size (MB)", "Latency (ms/sample)"])



# Print the comparison table nicely formatted

# float_format=lambda x: f"{x:.2f}" means all numbers will be shown
# with 2 decimal places for neatness.
print("\n=== Quantization Comparison Summary ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))



# Explain what each column means

print("\n Explanation of Columns:")
print("• Accuracy (%) – How correct the model's predictions are.")
print("• Size (MB) – How much memory/storage the model file takes.")
print("• Latency (ms/sample) – How fast each prediction runs (lower = faster).")



#  conclusion message(fix this to refelct new chnages on the summary table)

print("\n Conclusion:")

print("\nThese results demonstrate how each quantization method trades off between model size, speed, and accuracy.")
print("\nPTQ and GPTQ achieve near-baseline accuracy while greatly reducing model size and latency, making them strong options for deployment on limited-resource or edge devices.")
print("\nQAT accuracy is lower in this run because real int8 operations are not supported on macOS; it is expected to improve when tested on Béluga with true GPU-based quantization backends.")

print("\n Test run completed successfully on CPU (local test version).")
