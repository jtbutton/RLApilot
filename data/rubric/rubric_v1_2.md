---
type: ProjectNote
title: Rubric Ontology v1.2
description: null
tags: []
coverImage: null
project: [Radiolab Atlas]
task: [Draft Rubric Ontology v1.2]
---

Below is **Tagging Rubric v1.2**, fully revised using the insights from the HPGe pilot ingestion.
This version integrates **all recommended improvements**: stronger Scenario and Role criteria, concept granularity controls, category-confidence gating, and Resource deduplication guidance.

It remains strict, deterministic, hallucination-resistant, and fully aligned with ontology.json.

It is ready to replace v1.1 in your pipeline.

---

# **Radiolab Atlas – Tagging Rubric v1.2 (Refined After Pilot Test)**

### *(Strict, enumerated, ambiguity-minimized, ontology-aligned)*

---

# **0. GENERAL RULES (NO CHANGE TO CORE PRINCIPLES)**

The model **must not invent**:

- node types

- fields

- enum values

- relationship types

- concepts not grounded in the text

- roles, programs, scenarios, or competencies without explicit support

If information is not clearly present → **omit** it.

---

# **1. INPUT → OUTPUT CONTRACT**

Input:

- `chunk_text`

- `chunk_metadata`

- optional `ontology_summary`

Output:

```json
{
  "nodes": [],
  "relationships": [],
  "candidate_concepts": [],
  "confidence": 0.0,
  "notes": ""
}
```

No additional fields.

---

# **2. ALLOWED NODE TYPES (EXACT)**

The only allowable node types are:

1. `Resource`

2. `Concept`

3. `Competency`

4. `Scenario`

5. `Role`

6. `NetworkProgram`

7. `Instrument`

8. `CandidateConcept`

If no classification applies → return no nodes.

---

# **3. NODE CLASSIFICATION DECISION TREE (UPDATED)**

Follow in strict order.
Only proceed to later steps if earlier steps do not apply.

---

## **3.1 Resource Node (Unchanged — Works Well)**

Create `Resource` **only if** the chunk explicitly refers to a document-like object:

- SOP

- Method

- Guide

- Training Module

- Checklist

- Policy

- Standard

- Template

- Tool

- Calculator

- Report

- “This document / this procedure”

If none apply → **no Resource node**.

**Deduplication guidance:**
Use **identical title strings** for the same document across chunks.
The pipeline will merge duplicates.

---

## **3.2 Concept Node (Refined Granularity Rule)**

Create a `Concept` **only if**:

1. The chunk defines or explains a technical idea **AND**

2. The idea has standalone meaning beyond a minor detail

**Do NOT create a Concept when the text describes only:**

- A small QC parameter

- A minor physics detail

- Single-step analytical trick

- A footnote-level nuance

Instead, fold micro-details into larger concepts.

---

## **3.3 Competency Node (No Change)**

Create a `Competency` only if the chunk **explicitly describes**:

- A required skill

- A required ability

- A qualification for a role

Generic statements (“analysts must ensure calibration”) are **not sufficient**.

---

## **3.4 Scenario Node (NEW: Multi-Signal Threshold)**

Create a `Scenario` only if the chunk has **at least two** of:

1. **Actors**
(“technician”, “patient”, “radiographer”, “fire department”, etc.)

2. **Operational context or setting**
(industrial site, hospital, CRC, emergency zone)

3. **Sequence of actions**
(what happened first, next, last)

4. **Cause-and-effect**
(why exposure occurred)

5. **Consequences**
(detected dose, clinical outcomes)

6. **Equipment usage in context**
(source fell, radiography unit failure)

7. **Temporal or narrative structure**

If fewer than two → **no Scenario**.

This requirement prevents scenario inflation.

---

## **3.5 Role Node (NEW: Explicit Signal Requirement)**

Create a `Role` only if:

- The text *explicitly names a role*
(“Operations Chief”, “Analyst”, “Supervisor”, “Radiographer”), **AND**

- The chunk describes its responsibilities, authority, or required tasks.

Implicit references (“the analyst performs measurements”) → **do NOT create Role**.

---

## **3.6 NetworkProgram Node (No Change)**

Create a `NetworkProgram` only if:

- The program name is explicitly stated

- AND it is described in some meaningful way

Otherwise → omit.

---

## **3.7 Instrument Node (No Change)**

Create an `Instrument` only if:

- An instrument name appears explicitly

- It is described in technical terms

Modality-only references (“gamma measurement”) → **do not create Instrument**.

---

## **3.8 CandidateConcept (No Change to Criteria, Clarification Added)**

Create a `CandidateConcept` only if:

- The concept is clearly present in the text

- AND does **not** fit available concept categories

- AND is not represented by an existing Concept

CandidateConcept is a last resort.

---

# **4. NODE PROPERTIES (STRUCTURE UNCHANGED; CATEGORY RULE REFINED)**

### **NEW RULE:**

**Assign a concept category only if ≥70% confident.**
If category confidence is low → omit the category field.

This prevents incorrect category assignments.

---

# **5. RELATIONSHIP SELECTION (UNCHANGED)**

Allowed relationship types (exact):

```text
TEACHES
SUPPORTS
PREREQUISITE_FOR
RELATED_TO
REQUIRED_FOR
NEEDED_FOR
APPLIES_TO
OWNED_BY
DEVELOPS
ILLUSTRATES
CANDIDATE_FOR
USED_IN
REQUIRES_INSTRUMENT_COMPETENCY
DEPLOYED_IN
EMBODIES
```

Only create relationships if the text **clearly supports** them.

No inference.

---

# **6. UPDATED RELATIONSHIP DECISION MATRIX**

### **6.1 Resource → Concept / Competency / Scenario**

- TEACHES = Resource explains or defines the Concept

- SUPPORTS = Resource helps develop Competency

- APPLIES_TO = Resource is used within a Scenario

### **6.2 Program Ownership**

- OWNED_BY = Resource belongs to a NetworkProgram

### **6.3 Concept Relations**

- PREREQUISITE_FOR = explicit ordering

- RELATED_TO = explicit link

- REQUIRED_FOR = Concept → Competency

### **6.4 Competency → Role**

- NEEDED_FOR = Competency required to perform a Role

### **6.5 Instrument → Others**

- USED_IN = Instrument used in Resource

- REQUIRES_INSTRUMENT_COMPETENCY

- DEPLOYED_IN = in Scenario

- EMBODIES = Instrument represents Concept

### **6.6 CandidateConcept**

- CANDIDATE_FOR = candidate concept maps onto existing Concept

No other relations allowed.

---

# **7. CONFIDENCE RULES**

- ≥0.80 = strong

- 0.60–0.79 = moderate

- <0.60 = low / ambiguous

**Lower the score** if Scenario or Role signals are incomplete.

---

# **8. NOTES FIELD**

Short explanation of uncertainties or omissions.

---

# **9. EXAMPLE OUTPUT (NO CHANGE)**

(Same as in v1.1, omitted here for brevity.)

---

# 🎯 **Summary: What Changed from v1.1 → v1.2**

| Area                    | v1.1                   | v1.2 Improvement                                          |
| :---------------------- | :--------------------- | :-------------------------------------------------------- |
| Scenario Detection      | Single threshold       | **Two-signal requirement**                                |
| Role Detection          | Implicit               | **Explicit role-definition requirement**                  |
| Concept Granularity     | Allowed micro-concepts | **Guardrail against over-fragmentation**                  |
| Category Assignment     | Always required        | **≥70% confidence threshold**                             |
| Resource Consistency    | Implicit               | **Explicit deduplication guidance**                       |
| Independence of Classes | High                   | Even stronger separation of Roles/Scenarios from Concepts |

---

# 🚀 **v1.2 is now stable and ready for:**

- ingestion testing

- pipeline integration

- cross-document stress testing

- next-phase ontology refinement

If you'd like, I can now:

### → Re-run the HPGe pilot using v1.2 to show differences

[HPGe pilot data output](https://app.capacities.io/32569575-e8bb-48c3-87c0-166cdf4ce4ad/e21d2e51-4b55-498a-a8fb-c978f4f425b7)


### → Produce a test suite of positive/negative classification examples

### → Begin drafting the v2 ontology adjustments this pilot foreshadows

Which one should we do next?

