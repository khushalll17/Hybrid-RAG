# Hybrid RAG Travel Agent

## Overview
This project implements a **Hybrid Retrieval-Augmented Generation (RAG) Travel Agent** that answers travel booking queries using a strict separation of responsibilities between deterministic code and an LLM.

The system is intentionally designed so that:
- **All decisions, pricing, filtering, and calculations are handled deterministically in Python**
- **Retrieval (RAG) is used only for unstructured package information**
- **The LLM is used strictly for explanation and presentation**

This architecture prevents hallucinations, enforces business rules, and mirrors how real-world, production-grade AI systems are built.

---

## Key Design Principle: Hybrid RAG
Unlike traditional RAG systems where the LLM is allowed to reason, calculate, and decide, this system follows a **hybrid approach**:

- Deterministic code controls:
  - Flight selection
  - Time-based filtering (morning / afternoon / evening / night)
  - Cheapest flight logic
  - Package tier selection
  - Price extraction
  - Cost calculations
- RAG is used only to retrieve **relevant package descriptions and inclusions**
- The LLM:
  - Never performs calculations
  - Never modifies prices
  - Never applies constraints
  - Only explains results using provided context

This ensures **accuracy, reliability, and explainability**.


## High-Level System Flow

```mermaid
flowchart TD
    A[User Query] --> B[Query Parsing]
    B --> C[Deterministic Flight Selection]
    C --> D[Flight Data]

    A --> E[RAG Retrieval]
    E --> F[FAISS Vector Store]
    F --> G[Relevant Package Context]

    D --> H[Cost Calculation Engine]
    G --> H

    H --> I[Prompt Construction]
    I --> J[LLM Explanation Output]