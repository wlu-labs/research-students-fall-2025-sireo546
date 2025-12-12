# qat_demo.py
# Basic Quantization Aware Training demo
# ------------------------------
# Quantization Aware Training (QAT) demo in PyTorch
# This script shows the full process:
#   1. Build a small baseline neural net in FP32
#   2. Attach a quantization configuration (int8)
#   3. Prepare the model for QAT (adds fake quantization ops)
#   4. Train briefly on dummy data
#   5. Convert the model into a real int8 version
# ------------------------------


import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"
else:
    torch.backends.quantized.engine = "fbgemm"

# Small prototype network
# ------------------------------
# Step 1: Define a simple model
# ------------------------------
# This is just a tiny fully connected network to practice QAT.
# Input is 28x28 (like MNIST images), one hidden layer of 128 units,
# ReLU activation, then an output layer of 10 classes.
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten (28x28 → 784)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# create the model in default FP32 precision
model = SimpleNet()



# ------------------------------
# Step 2: Attach a QAT config
# ------------------------------
# qconfig tells PyTorch which quantization scheme to simulate.
# "fbgemm" is the backend for int8 quantization on CPUs.
model.qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)




# ------------------------------
# Step 3: Prepare for QAT
# ------------------------------
# prepare_qat inserts "fake quantization" and "fake dequantization"
# layers into the model. These act like quantization during forward
# passes (simulate rounding/clamping to int8), but still allow gradients
# to flow in FP32 for training.
model = prepare_qat(model)



# ------------------------------
# Step 4: Train briefly
# ------------------------------
# Instead of real images, we use random dummy data.
# The point here is to show the QAT process, not accuracy.
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()



print("Training with fake quantization...")
for epoch in range(3): # just 3 epochs for demo speed
    # batch of 16 fake grayscale "images"
    inputs = torch.randn(16, 1, 28, 28)         # fake images
   
    # random labels (10 classes)
    labels = torch.randint(0, 10, (16,))      # fake labels
   
    # forward pass (fake quantization applied here)
    outputs = model(inputs)
   
    # compute loss
    loss = loss_fn(outputs, labels)


  # backward pass + weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# ------------------------------
# Step 5: Convert to real int8
# ------------------------------
# Now we replace the fake quantization layers with real int8
# operations. This produces a true quantized model that runs
# faster and uses less memory.
model.eval()
quantized_model = convert(model)
print("\nConverted to INT8 model:")
print(quantized_model)
