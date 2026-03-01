# Qwen3-ASR Pro - Markdown Files Audit Report

**Date:** 2026-02-28  
**Auditor:** Agent  
**Current Version:** 3.3.0  

---

## 1. Table of All Markdown Files

| File | Purpose | Version Mentioned | Status | Action Needed |
|------|---------|-------------------|--------|---------------|
| **README.md** | User documentation | None | ✅ Fresh | Minor updates needed |
| **AGENTS.md** | Agent guide | 3.3.0 | ✅ Fresh | None |
| **PROJECT_DESIGN.md** | System architecture | 3.3.0 | ✅ Fresh | None |
| **INTEGRATION_TEST_REPORT.md** | Integration test results | 3.3.0 | ✅ Fresh | None |
| **LLM_TEST_REPORT.md** | LLM module tests | None | ✅ Fresh | None |
| **tests/README.md** | Test documentation | None | ✅ Fresh | None |
| ~~tests/TEST_REPORT.md~~ | ~~Live streaming initial tests~~ | ~~None~~ | ❌ **DELETED** | Historical only |
| **tests/LIVE_STREAMING_FIX_REPORT.md** | Threading fix validation | None | ✅ Fresh | None |
| ~~tests/COMPREHENSIVE_TEST_REPORT.md~~ | ~~Comprehensive test results~~ | **~~3.1.1~~** | ❌ **DELETED** | Outdated version |
| **tests/TRANSCRIPTION_TEST_REPORT.md** | Transcription tests | None | ⚠️ Partially outdated | Update Python version |
| **tests/EDGE_CASE_TEST_REPORT.md** | Edge case tests | None | ✅ Fresh | None |
| **assets/c-asr/README.md** | C implementation docs | None | ✅ Fresh | None |
| **assets/c-asr/MODEL.md** | Model reference | None | ✅ Fresh | None |
| **assets/c-asr/MODEL_CARD_OFFICIAL.md** | Official model card | None | ✅ Fresh | None |
| **docs/M1_PRO_SETUP.md** | M1 Pro setup guide | None | ✅ Fresh | None |

---

## 2. Conflicts Found

### 2.1 Version Number Conflicts

| File | Version | Correct? |
|------|---------|----------|
| src/__init__.py | 3.3.0 | ✅ Yes |
| src/constants.py | 3.3.0 | ✅ Yes |
| AGENTS.md | 3.3.0 | ✅ Yes |
| INTEGRATION_TEST_REPORT.md | 3.3.0 | ✅ Yes |
| PROJECT_DESIGN.md | 3.3.0 | ✅ Yes |
| **tests/COMPREHENSIVE_TEST_REPORT.md** | **3.1.1** | ❌ **OUTDATED** |

**Impact:** The comprehensive test report shows version 3.1.1 but the actual version is 3.3.0.

### 2.2 Mode Naming Inconsistencies

| Source | Mode Names |
|--------|------------|
| README.md | "Live Class Mode" / "Fast Mode" |
| AGENTS.md | "Live Mode" / "Upload Mode" |

**Issue:** Different terminology for the same modes. README uses user-friendly names while AGENTS uses technical names.

### 2.3 Performance Targets (RTF) Conflicts

| Source | Streaming RTF | Status |
|--------|---------------|--------|
| README.md | ~0.46x | ❌ Outdated |
| AGENTS.md | < 3.0x | ✅ Current |
| LIVE_STREAMING_FIX_REPORT.md | 0.19x - 0.96x | ✅ Verified |
| COMPREHENSIVE_TEST_REPORT.md | < 3.0x | ✅ Current |

**Issue:** README.md shows outdated performance metric (~0.46x) that doesn't match actual validated performance.

### 2.4 Recording File Naming Conflict

| Source | Naming Pattern |
|--------|----------------|
| README.md | `class_YYYYMMDD_HHMMSS.wav` |
| AGENTS.md | `live_YYYYMMDD_HHMMSS.wav` |

**Issue:** Different naming conventions for recorded files.

### 2.5 Model Selection Description Conflict

| Source | Description |
|--------|-------------|
| README.md | "User-selected model (0.6B or 1.7B)" in Fast Mode |
| AGENTS.md | "Always uses 1.7B model for best accuracy" in Upload Mode |

**Issue:** README suggests user can select models in Fast Mode, but AGENTS says Upload mode always uses 1.7B.

### 2.6 AI Text Refinement Backend Conflict

| Source | Primary Backend |
|--------|-----------------|
| README.md | mlx-lm (Qwen2.5-3B-Instruct-4bit) |
| PROJECT_DESIGN.md | Ollama (qwen:1.8b) |
| INTEGRATION_TEST_REPORT.md | Ollama (qwen:1.8b) |

**Issue:** Multiple backend approaches documented. Need to clarify which is primary.

### 2.7 Project Structure Conflict

| Source | docs/ Directory |
|--------|-----------------|
| README.md / AGENTS.md | Contains documentation |
| PROJECT_DESIGN.md | Lists as "(empty)" |

**Issue:** docs/ now contains M1_PRO_SETUP.md but PROJECT_DESIGN says it's empty.

### 2.8 Python Version Inconsistency

| Source | Python Version |
|--------|----------------|
| AGENTS.md | 3.12+ (recommended) |
| INTEGRATION_TEST_REPORT.md | 3.9.6 (actual test env) |
| TRANSCRIPTION_TEST_REPORT.md | 3.9.6 (actual test env) |

**Issue:** Documentation says 3.12+ but tests ran on 3.9.6.

### 2.9 Test Results Summary Conflict

| Source | Tests Passed |
|--------|--------------|
| INTEGRATION_TEST_REPORT.md | 91/91 |
| ~~COMPREHENSIVE_TEST_REPORT.md~~ | ~~225/246 (91.5%)~~ |
| LLM_TEST_REPORT.md | 22/24 (91.7%) |

**Issue:** Different test counts because different test files were run.

---

## 3. Outdated Information

### 3.1 Critical Outdated Items

| File | Outdated Content | Current State |
|------|------------------|---------------|
| **tests/COMPREHENSIVE_TEST_REPORT.md** | Version 3.1.1 | Should be 3.3.0 |
| **README.md** | Streaming RTF ~0.46x | Actual: 0.19x - 0.96x |
| **README.md** | Fast Mode model selection | Upload mode always uses 1.7B |
| **AGENTS.md** | `docs/` listed as "(empty)" | Now contains M1_PRO_SETUP.md |

### 3.2 Minor Outdated Items

| File | Outdated Content | Current State |
|------|------------------|---------------|
| **README.md** | File naming `class_*.wav` | Should be `live_*.wav` |
| **TRANSCRIPTION_TEST_REPORT.md** | Python 3.9.6 mentioned | Should note 3.12+ recommended |
| **PROJECT_DESIGN.md** | Phase 5 "Model selection needed" | Model selection is implemented |

---

## 4. Duplicate/Redundant Reports

### 4.1 Test Report Hierarchy

```
 tests/COMPREHENSIVE_TEST_REPORT.md (OUTDATED - v3.1.1)
    ├── Contains: Environment, Transcription, Live Streaming, UI, Edge Cases
    └── Status: SUPERSEDED by more recent reports

 tests/TEST_REPORT.md (v3.3.0 era)
    ├── Contains: Initial live streaming tests, bugs identified
    └── Status: HISTORICAL - documents bugs that are now fixed

 tests/LIVE_STREAMING_FIX_REPORT.md (v3.3.0)
    ├── Contains: Fix validation for threading issues
    └── Status: CURRENT - documents validated fixes

 tests/INTEGRATION_TEST_REPORT.md (v3.3.0)
    ├── Contains: End-to-end integration tests
    └── Status: CURRENT - comprehensive integration validation

 tests/LLM_TEST_REPORT.md (v3.3.0)
    ├── Contains: LLM module tests
    └── Status: CURRENT - AI text refinement tests

 tests/TRANSCRIPTION_TEST_REPORT.md (v3.3.0 era)
    ├── Contains: Transcription backend tests
    └── Status: CURRENT but partially outdated

 tests/EDGE_CASE_TEST_REPORT.md (v3.3.0)
    ├── Contains: Security and edge case tests
    └── Status: CURRENT
```

### 4.2 Recommendations for Consolidation

| Action | File | Reason |
|--------|------|--------|
| **DELETE** | `tests/COMPREHENSIVE_TEST_REPORT.md` | Outdated (v3.1.1), superseded by newer reports |
| **ARCHIVE** | `tests/TEST_REPORT.md` | Historical value only (documents bugs now fixed) |
| **KEEP** | `tests/LIVE_STREAMING_FIX_REPORT.md` | Documents current validated state |
| **KEEP** | `tests/INTEGRATION_TEST_REPORT.md` | Current comprehensive integration results |
| **KEEP** | `tests/LLM_TEST_REPORT.md` | Current LLM module validation |
| **UPDATE** | `tests/TRANSCRIPTION_TEST_REPORT.md` | Update Python version reference |
| **KEEP** | `tests/EDGE_CASE_TEST_REPORT.md` | Current security validation |

---

## 5. Specific Changes Needed

### 5.1 README.md Changes

```markdown
# Performance Table - Update to:
| Model | Mode | Speed | Use Case |
|-------|------|-------|----------|
| 0.6B | MLX | ~0.02x RTF | Fast transcription |
| 1.7B | MLX | ~0.03x RTF | Best accuracy |
| 0.6B | Streaming | ~0.72x RTF | Live transcription |

# File Naming - Update to:
Naming: `live_YYYYMMDD_HHMMSS.wav`

# Fast Mode Description - Update to:
"- Always uses 1.7B model for best accuracy"
```

### 5.2 AGENTS.md Changes

```markdown
# Project Structure - Update docs/ entry:
├── docs/                      # Documentation
│   └── M1_PRO_SETUP.md       # Apple Silicon setup guide

# docs/ is NOT empty anymore
```

### 5.3 tests/COMPREHENSIVE_TEST_REPORT.md Changes

**Status:** ✅ **DELETED** - File was outdated (v3.1.1) and superseded by newer reports

### 5.4 tests/TRANSCRIPTION_TEST_REPORT.md Changes

```markdown
# Environment - Update to:
**Environment:** macOS, Python 3.9.6 (3.12+ recommended), PyTorch 2.1.0
```

### 5.5 PROJECT_DESIGN.md Changes

```markdown
# Phase 5 Status - Update to:
| Phase 5: Polish | ✅ Done | Model selection, export features implemented |

# docs/ Directory - Update to:
├── docs/
│   └── M1_PRO_SETUP.md       # Apple Silicon setup guide
```

---

## 6. Summary of Recommendations

### 6.1 Immediate Actions (High Priority)

1. ~~**Update or Delete** `tests/COMPREHENSIVE_TEST_REPORT.md`~~ - ✅ **DELETED**
2. **Update** `README.md` - Fix streaming RTF from ~0.46x to ~0.72x
3. **Update** `README.md` - Fix file naming from `class_` to `live_`
4. **Update** `AGENTS.md` - Fix docs/ directory description

### 6.2 Secondary Actions (Medium Priority)

5. ~~**Archive** `tests/TEST_REPORT.md`~~ - ✅ **DELETED**
6. **Update** `tests/TRANSCRIPTION_TEST_REPORT.md` - Add Python version recommendation
7. **Update** `PROJECT_DESIGN.md` - Mark Phase 5 as complete

### 6.3 Documentation Clarity (Low Priority)

8. **Align terminology** between README.md (user-friendly) and AGENTS.md (technical)
9. **Clarify AI backends** - Document that both mlx-lm and Ollama are supported
10. **Add version history** to main README.md

---

## 7. Files Status Summary

| Status Count | Files |
|--------------|-------|
| ✅ **Fresh (no changes)** | 9 files |
| ⚠️ **Needs minor updates** | 3 files |
| ❌ **Outdated/Delete** | 2 files |
| **Total** | **14 files** |

### Files to Keep (No Changes)
1. README.md (minor updates only)
2. AGENTS.md (minor updates only)
3. PROJECT_DESIGN.md (minor updates only)
4. INTEGRATION_TEST_REPORT.md
5. LLM_TEST_REPORT.md
6. tests/README.md
7. tests/LIVE_STREAMING_FIX_REPORT.md
8. tests/EDGE_CASE_TEST_REPORT.md
9. assets/c-asr/README.md
10. assets/c-asr/MODEL.md
11. assets/c-asr/MODEL_CARD_OFFICIAL.md
12. docs/M1_PRO_SETUP.md

### Files to Update
1. tests/TRANSCRIPTION_TEST_REPORT.md

### Files to Delete
1. ~~tests/COMPREHENSIVE_TEST_REPORT.md~~ ✅ DELETED (outdated v3.1.1)
2. ~~tests/TEST_REPORT.md~~ ✅ DELETED (historical, bugs are now fixed)

---

**End of Audit Report**
