# Tensor Query Processor (TQP)

The **Tensor Query Processor (TQP)** transforms SQL queries into tensor programs optimized for execution on modern **Tensor Computation Runtimes (TCRs)** such as **PyTorch**. TQP supports both **single-node** and **distributed** execution, enabling high-performance analytics on CPUs, GPUs, and multi-GPU clusters.

---

## Overview

TQP operates through two major phases:

| Phase           | Description                         |
| --------------- | ----------------------------------- |
| **Compilation** | SQL → Tensor program generation     |
| **Execution**   | Convert and process data as tensors |

This enables TQP to run full analytical workloads—including TPC-H—using purely tensor APIs.

---

# Single-Node TQP Architecture

The single-node system is built on two foundations:
✅ Columnar tensor-based data representation  
✅ A four-layer compiler pipeline mapping relational operators → tensor programs

---

## A. Data Representation

Relational tables are converted into columnar tensor format:

| Data Type                   | Tensor Representation                                |
| --------------------------- | ---------------------------------------------------- |
| Numeric (int/float/decimal) | `n×1` tensors (decimal cast to float)                |
| Date                        | `n×1` numeric tensor storing nanoseconds since epoch |
| String                      | `n×m` tensor of padded character codes               |

Each column = one tensor → optimized for TCRs and vectorized execution.

---

## B. Query Compilation Pipeline

| Layer                               | Role                                                               |
| ----------------------------------- | ------------------------------------------------------------------ |
| **Parsing**                         | SQL → IR Graph using frontend system (e.g., Spark Catalyst)        |
| **Canonicalization + Optimization** | IR-to-IR rewrites to normalize semantics & improve performance     |
| **Planning**                        | IR Graph → **tensor operator plan** mapped into PyTorch programs   |
| **Execution**                       | Operator plan wrapped into executor (PyTorch / TorchScript / ONNX) |

Expression trees (e.g., `price * qty`) are recursively mapped to PyTorch ops (e.g. `torch.mul`).

---

## C. Relational Operators Using Tensor APIs

All implementations strictly rely on existing TCR tensor ops:

| Operator                  | Tensor-Based Strategy                                    |
| ------------------------- | -------------------------------------------------------- |
| **Filter**                | `torch.lt` → boolean mask → `masked_select`              |
| **Sort-Join (Equi-Join)** | Sort keys → histograms (`bincount`) → `bucketize` search |
| **Hash-Join**             | Hashing using index tensors + `scatter_` + probing loops |
| **Group-By Aggregation**  | Sort group keys → `uniqueConsecutive` → indexed compute  |

These enable late materialization and massive GPU parallelism.

---

# Distributed TQP Architecture

TQP extends to a **data-parallel multi-GPU system**:

✅ Each GPU runs an independent TQP instance  
✅ Partition-local computation  
✅ Data exchange inserted automatically by compiler

---

## A. Computational Model

| Step              | Description                                     |
| ----------------- | ----------------------------------------------- |
| Data Partitioning | Row partitions distributed across GPUs          |
| Local Execution   | Each GPU processes its partitions independently |
| MPI Launch        | All TQP nodes executed as MPI ranks             |

---

## B. Automatic Data Exchange Insertion

Compiler injects exchange operators where relational semantics require:

- Before joins
- Before group-by
- Final aggregation merging

---

## C. High-Performance Data Movement

Uses **HPC collective libraries**:

| Operation         | Library APIs              | Notes                                      |
| ----------------- | ------------------------- | ------------------------------------------ |
| **Shuffle**       | `ncclSend` + `ncclRecv`   | N² messages, handles skew & variable sizes |
| **Broadcast**     | `ncclBroadcast`           | Efficient one→many                         |
| **Aggregation**   | `ncclAllReduce`           | Fast distributed reductions                |
| **Size Exchange** | Lightweight metadata sync | Enables correct buffer allocation          |

Optimized for NVLink / InfiniBand bandwidth.

---

# Summary

| Feature                         | Supported |
| ------------------------------- | :-------: |
| SQL → tensor compilation        |           |
| Full relational algebra         |           |
| Support for TPC-H               |           |
| Multi-GPU execution             |           |
| TCR portability via PyTorch ops |           |
| Export: TorchScript / ONNX      |           |

---

## 📌 Key Benefits

- Runtime built on ML tensor cores → massive parallelism
- Zero rewrites needed to support multiple hardware vendors
- Unified performance posture with deep-learning workloads

---

### Future Work

- Dictionary-encoded string tensors
- Cost-based distributed planning
- Adaptive exchange strategies for skew-heavy data
