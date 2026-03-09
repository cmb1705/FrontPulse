# Validation Strategy: Early Detection of Paradigm Shifts

## Goal

Determine if our community detection pipeline can identify emerging research fronts within 1-3 quarters of landmark/paradigm-shifting papers being published.

## Resolution Selection

**Selected: 0.001**

- Rationale: Optimal balance between early detection sensitivity and tracking stability
- Community sizes: ~75 papers mean, ~39 median
- Growth responsive: 1750% growth rate shows system adapts to field evolution
- Coupling edges (beta=0.3) enable detection of emerging clusters before direct citations form

## Validation Protocol

### Step 1: Identify Landmark Papers

Select 2-3 known paradigm-shifting papers in the corpus field (ideally papers where there is consensus they were transformative).

### Step 2: Locate Publication Quarters

For each landmark paper:

- Note publication quarter: Q_landmark
- Note the paper's work_id

### Step 3: Trace Community Formation

For each landmark paper:

1. Check which community (if any) contains the landmark paper at Q_landmark
2. Track forward quarter-by-quarter until the paper appears in a distinct community
3. Record Q_detected = first quarter where the landmark paper is in a community with distinct identity (not absorbed in larger established cluster)

### Step 4: Calculate Detection Lag

For each landmark paper:

- Detection lag = Q_detected - Q_landmark (in quarters)

### Success Criteria

- **Excellent:** Average lag ≤ 3 quarters (detection within 9 months)
- **Good:** Average lag ≤ 5 quarters (detection within 15 months)
- **Acceptable:** Average lag ≤ 8 quarters (detection within 2 years)
- **Needs tuning:** Average lag > 8 quarters

### Step 5: Adjust if Needed

If detection lag exceeds acceptable threshold:

- Consider testing resolution = 0.005 for higher sensitivity
- Check if coupling parameters need adjustment
- Examine split/merge events to see if clusters are forming but not separating

## Expected Behavior

### Early Detection Mechanism (Q0-Q3)

- **Q0:** Landmark paper published
- **Q1-Q2:** First wave of papers cite landmark + shared foundational references
  - Coupling edges (shared references) create initial cluster cohesion
  - At resolution 0.001, cluster may separate if ≥15-20 tightly coupled papers
- **Q3:** Cluster grows to 30-50 papers, becomes clearly distinct

### Limitations

- Cannot detect before publications appear (obvious but worth stating)
- Truly novel fields with no precedent may take longer (fewer shared references initially)
- Slow-burn paradigms may need 4-6 quarters to reach critical mass for detection

## Notes from Resolution Sweep

- Resolution 0.0005: Too coarse, emerging clusters absorbed until ~50-80 papers
- Resolution 0.001: Sweet spot for detection at 20-50 paper cluster size
- Resolution 0.005: Too sensitive, high noise/false positives (PIA rate spike)
- Resolution 0.01: Over-fragmentation, unstable communities

## Date

Created: 2025-10-29
Sweep data: `data/out/resolution_sweep_cumulative.json` (2000Q1-2017Q4)
