# GloVe Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CROWN and LIME run when `torchtext` is unavailable by falling back to local GloVe text files.

**Architecture:** Add one shared loader at the repo root that lazily tries `torchtext` and falls back to streaming `glove/*.txt`. Update both corpus preprocessors to use the shared loader instead of importing `torchtext` at module import time.

**Tech Stack:** Python, PyTorch, unittest

---

### Task 1: Add shared GloVe fallback loader

**Files:**
- Create: `D:\MIND\glove_loader.py`
- Test: `D:\MIND\tests\test_glove_fallback.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Wire CROWN and LIME to the shared loader

**Files:**
- Modify: `D:\MIND\crown-www25\corpus.py`
- Modify: `D:\MIND\lime-cikm25\corpus.py`
- Test: `D:\MIND\tests\test_large_training_entrypoints.py`

- [ ] **Step 1: Remove module-level `torchtext` import**
- [ ] **Step 2: Replace inline GloVe loading with shared loader**
- [ ] **Step 3: Run targeted tests and syntax checks**
