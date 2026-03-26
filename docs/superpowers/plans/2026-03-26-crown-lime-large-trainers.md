# CROWN And LIME Large Trainers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add root-level CROWN/LIME training entrypoints that default to the local `dataset/MINDlarge_*` layout, write validation metrics into `results`, and emit Codabench-compatible `prediction.txt` plus zip artifacts.

**Architecture:** Keep the original paper implementations intact where possible, and add thin root wrappers plus small config/output fixes inside each upstream project. Resolve dataset roots relative to the repository instead of absolute paths so the same code works locally and on a server checkout.

**Tech Stack:** Python, PyTorch, unittest, existing CROWN/LIME codebase

---

### Task 1: Lock Large-Dataset Path And Output Requirements With Tests

**Files:**
- Create: `D:\MIND\tests\test_large_training_entrypoints.py`

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL because large dataset defaults and submission helpers are missing.

- [ ] **Step 3: Write minimal implementation**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`
Expected: PASS

### Task 2: Patch CROWN And LIME Internal Config/Output Logic

**Files:**
- Modify: `D:\MIND\crown-www25\config.py`
- Modify: `D:\MIND\lime-cikm25\config.py`
- Modify: `D:\MIND\crown-www25\util.py`
- Modify: `D:\MIND\lime-cikm25\util.py`

- [ ] **Step 1: Extend configs to accept `large` and resolve paths relative to repo root**
- [ ] **Step 2: Ensure prediction output uses `prediction.txt` and zip packaging for large test**
- [ ] **Step 3: Preserve dev metrics logging into `results`**
- [ ] **Step 4: Run targeted tests**

### Task 3: Add Root-Level Training Entrypoints

**Files:**
- Create: `D:\MIND\train_crown.py`
- Create: `D:\MIND\train_lime.py`

- [ ] **Step 1: Add wrappers that default to `large` and current repo-relative dataset paths**
- [ ] **Step 2: Run wrappers in `--help` / dry-start mode to verify imports and argument wiring**

### Task 4: Verify End-To-End Commands

**Files:**
- Modify: `D:\MIND\README.md`

- [ ] **Step 1: Run verification commands for tests and wrapper startup**
- [ ] **Step 2: Document exact train/test submission commands**
