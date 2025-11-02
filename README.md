# 🐦 Phoenix Recovery: Robust PyTorch Staged Execution Framework

## 🛡️ Project Overview: Crash-Resilient ML Infrastructure

This project implements a low-level structural wrapper around standard PyTorch neural networks to create a highly **crash-resilient and checkpointable** training environment. Named "Phoenix Recovery" after its ability to restart and resume after failure, the framework is designed to prevent catastrophic data loss and maximize efficiency during long-running or volatile deep learning training pipelines.

It leverages core systems engineering principles (robustness, fault tolerance) and applies them directly to the PyTorch computational graph.

## 💡 Core Concepts

The framework is built on two primary components: `Stage` and `StagedModule`.

### 1. The `Stage` Module (`nn.Module` Wrapper)

The `Stage` class is a wrapper that encapsulates any standard PyTorch layer (e.g., `nn.Linear`, `nn.ReLU`). It defines a clear, segmentable boundary in the forward pass of the neural network, allowing for localized exception handling and detailed logging.

### 2. The `StagedModule` (Execution Manager)

This class acts as the main training orchestrator. It manages the epoch loop and implements the crash recovery logic. If an exception (a simulated or real crash) is thrown within any wrapped `Stage` during training, the `StagedModule` catches it and uses a defined `restartMethod` to gracefully resume the process, often by reloading the last successful state or re-initializing the current epoch.

## 🚀 Key Features

* **Fault Tolerance:** Implements graceful recovery logic to prevent complete job failure due to transient errors or hardware issues.

* **Modular Logging:** Provides clear points within the network execution for logging errors, allowing developers to pinpoint the failing layer or stage.

* **Training Reliability:** Significantly increases the reliability of training runs for models that take days or weeks to converge.

* **Systems Integration:** Bridges low-level exception handling techniques with high-level ML framework execution.

## 💻 Implementation Details & Execution

The core implementation uses custom wrapper classes to intercept exceptions and an execution manager to handle the restart loop. The `DummyLayer` is used to simulate sporadic, non-deterministic failures for testing the recovery mechanism.

### Prerequisites

This framework requires a standard PyTorch environment:

* **Python:** 3.8+
* **PyTorch:** Latest stable version
* **Other:** `typing`, `random` (standard Python modules)

### Running the Recovery Demo

Assuming the framework and model are saved in a runnable script (e.g., `train.py`), the project can be executed from the command line:

```bash
python model.py
