# Simulated Examples for TFNBS Benchmarking

Critical analysis of synthetic datasets from `sim_datasets.py` and `fbc_superiority_tests.py` for method benchmarking in brain connectivity analysis.

## Overview

This document compares two simulation frameworks:
1. **sim_datasets.py**: Comprehensive benchmark suite with 8 scenarios
2. **fbc_superiority_tests.py**: Targeted tests for FBC-TFNBS advantages

---

## Part I: sim_datasets.py Scenarios

### S1: Single Edge Effect

**Motivation:** Test the hardest case for component-based methods - a focal effect on just one edge.

**Biological Plausibility:** ★★☆☆☆
- Single-edge effects are rare in neuroscience
- Most brain pathology affects distributed networks, not isolated connections
- Could represent: highly specific lesion, single fiber tract damage

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Power analysis | Useful floor test |
| FWER control | Good - tests edge-level specificity |
| Realistic | Poor - too artificial |

**Pros:**
- Clear ground truth
- Tests method sensitivity at the limit
- Exposes false positive patterns

**Cons:**
- Component-based methods (NBS, TFNBS) are NOT designed for this
- May unfairly penalize methods that assume distributed effects
- Not representative of real neuroimaging findings

**Recommendation:** Include as a **negative control** only. Methods should NOT excel here.

---

### S2: Within-Module Dense Block

**Motivation:** Test detection of effects concentrated within a functional module (e.g., all Visual-Visual connections).

**Biological Plausibility:** ★★★★★
- Highly realistic - many disorders affect specific functional systems
- Examples: Visual cortex in macular degeneration, motor network in Parkinson's
- Matches the "functional block" hypothesis of FBC-TFNBS

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Power analysis | Excellent |
| Clinical relevance | High |
| Method discrimination | Good for block-based vs topological |

**Pros:**
- Strong biological motivation
- Tests core assumption of network-based methods
- Clear expected winner hierarchy: FBC > NI-TFNBS > TFNBS (for small effects)

**Cons:**
- May be "too easy" if effect is strong
- Assumes perfect module assignment (unrealistic)
- Doesn't test cross-module effects

**Recommendation:** **Essential benchmark**. Core test case.

---

### S3: Within-Module Mixed Sign

**Motivation:** Test methods when effects have both increases AND decreases within the same module.

**Biological Plausibility:** ★★★★☆
- Realistic for reorganization/plasticity studies
- Example: Stroke recovery - some connections strengthen, others weaken
- Challenging for methods that assume unidirectional effects

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Method stress-test | Excellent |
| Biological relevance | Good |
| Interpretation | Complex |

**Pros:**
- Tests robustness to sign heterogeneity
- Exposes methods that only detect unidirectional effects
- Relevant for longitudinal/intervention studies

**Cons:**
- Ground truth interpretation is ambiguous
- Which edges "should" be detected?
- May penalize methods that correctly separate + and - effects

**Recommendation:** Include with **clear analysis protocol** - separate + and - analyses.

---

### S4: Tree Component

**Motivation:** Test acyclic connected structures (chains/pathways without loops).

**Biological Plausibility:** ★★★☆☆
- Partially realistic - some pathways are hierarchical (sensory processing)
- BUT: real brain networks have many cycles/loops
- Could represent: feedforward sensory cascade

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Topological testing | Good |
| Sparse component detection | Excellent |
| Biological validity | Moderate |

**Pros:**
- Tests sparse connected component detection
- Clear topological structure
- Good for comparing NBS vs TFNBS

**Cons:**
- Trees are rare in real brain networks (too sparse)
- Artificially favors methods that assume simple connectivity
- Ignores recurrent/feedback connections

**Recommendation:** Include as **topological stress test**, not as primary benchmark.

---

### S5: Cycle Component

**Motivation:** Test cyclic structures (closed loops).

**Biological Plausibility:** ★★★★☆
- More realistic than trees - brain has many recurrent loops
- Examples: Thalamocortical loops, cortico-basal ganglia circuits
- Represents feedback/regulatory systems

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Loop detection | Good |
| Contrast with tree | Useful |
| Realistic structure | Better than tree |

**Pros:**
- Tests cycle/loop detection capability
- More biologically plausible than trees
- Good contrast with S4

**Cons:**
- Still relatively sparse
- Pure cycles are also rare (usually embedded in denser structures)
- Limited edges make statistical detection hard

**Recommendation:** Include alongside S4 for **topological comparison**.

---

### S6: Distance-Decay with Long-Range Shortcuts

**Motivation:** Test detection of long-range connections in a spatially-organized null.

**Biological Plausibility:** ★★★★★
- Excellent - brain connectivity follows distance-decay + long-range hubs
- Models: Default Mode Network (long-range), rich-club connections
- Realistic null model (spatial autocorrelation)

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Spatial structure | Excellent |
| Long-range detection | Critical test |
| Null model realism | High |

**Pros:**
- Biologically motivated null model
- Tests detection of "interesting" long-range connections
- Exposes methods that over-penalize distance

**Cons:**
- Requires coordinate information (not always available)
- Distance-graded effect size may confound interpretation
- "Long-range" definition is arbitrary

**Recommendation:** **Essential benchmark** for methods claiming spatial sensitivity.

---

### S7: Multiple Disjoint Components

**Motivation:** Test FWER control when multiple true effects exist.

**Biological Plausibility:** ★★★☆☆
- Moderately realistic - some conditions affect multiple systems
- Example: Schizophrenia (prefrontal + temporal), ADHD (multiple networks)
- BUT: components being completely disjoint is unlikely

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| FWER stress-test | Excellent |
| Multiple testing | Critical |
| Sensitivity trade-off | Exposes it well |

**Pros:**
- Tests the multiple comparisons problem directly
- Reveals methods that over-correct (miss small effects)
- Good for heterogeneous effect sizes

**Cons:**
- Artificially clean separation
- Real multi-system effects usually share nodes
- Heterogeneous effects complicate ground truth definition

**Recommendation:** Include for **FWER validation**, but acknowledge artificiality.

---

### S8: Structural Prior (Aligned vs Misaligned)

**Motivation:** Test prior-informed methods when prior is correct vs incorrect.

**Biological Plausibility:** ★★★★☆
- Very relevant - structural connectivity informs functional
- "Aligned" = effect follows structural backbone
- "Misaligned" = effect in unexpected locations

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Prior validation | Critical |
| Robustness testing | Excellent |
| Method comparison | Reveals prior dependency |

**Pros:**
- Essential for validating prior-informed methods
- Tests both best-case and worst-case scenarios
- Clinically relevant (structural damage → functional change)

**Cons:**
- Requires structural connectivity data
- "Correct" prior is subjective
- Binary aligned/misaligned is oversimplified

**Recommendation:** **Essential for NI-TFNBS validation**. Include both conditions.

---

## Part II: fbc_superiority_tests.py Scenarios

### Scenario 1: Scattered Stars (Disconnected Hubs)

**Motivation:** Multiple hub nodes with spokes, but hubs don't connect - topologically disconnected but functionally related.

**Biological Plausibility:** ★★★★☆
- Realistic for multi-hub network effects
- Example: DMN has multiple hubs (mPFC, PCC, angular gyrus)
- Each hub's local connections affected, but hubs don't directly connect

**Comparison with sim_datasets:**
- NOT covered in sim_datasets.py
- Fills important gap: disconnected but functionally coherent

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| FBC advantage | Should be large |
| TFNBS weakness | Exposed |
| Biological validity | Good |

**Pros:**
- Perfect test case for FBC vs TFNBS
- Biologically motivated (multi-hub networks)
- Clear prediction: FBC > TFNBS

**Cons:**
- Artificially prevents hub-hub connections
- Real hub networks have some inter-hub connectivity
- May overstate FBC advantage

**Recommendation:** **Keep** - unique and valuable test case.

---

### Scenario 2: Checkerboard (No Shared Nodes)

**Motivation:** Edges selected to have NO shared nodes - maximally disconnected.

**Biological Plausibility:** ★★☆☆☆
- Unrealistic - brain effects almost always share nodes
- Artificial worst-case for topological methods
- No clear biological analog

**Comparison with sim_datasets:**
- Similar spirit to S1 (single edge) but more edges
- Extreme stress test

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Method stress-test | Excellent |
| Biological validity | Poor |
| FBC advantage | Maximum |

**Pros:**
- Clear theoretical advantage for FBC
- Tests edge-level vs component-level explicitly
- Useful lower bound for TFNBS

**Cons:**
- Too artificial for realistic benchmarking
- May mislead about real-world performance
- Selection process (greedy no-overlap) is contrived

**Recommendation:** Include as **theoretical extreme**, not primary benchmark.

---

### Scenario 3: Cross-Module Pathway

**Motivation:** Effect concentrated in between-module connections (e.g., Visual-Motor pathway).

**Biological Plausibility:** ★★★★★
- Highly realistic - many effects are in inter-network connections
- Example: Attention disorders (Salience-DMN), Motor learning (Motor-Cerebellar)
- Tests between-block detection

**Comparison with sim_datasets:**
- Complements S2 (within-module) from sim_datasets
- Fills gap: sim_datasets doesn't have explicit between-module scenario

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Between-module detection | Excellent |
| Clinical relevance | High |
| Method discrimination | Good |

**Pros:**
- Essential complement to within-module tests
- Biologically important
- Tests FBC's between-block handling

**Cons:**
- Edges still artificially selected (no shared nodes)
- Real pathways have some topological connection
- Single pathway may be too focused

**Recommendation:** **Essential benchmark** - add to sim_datasets.py.

---

### Scenario 4: Very Weak Diffuse Effect

**Motivation:** Test detection of very weak but widespread effects.

**Biological Plausibility:** ★★★★★
- Highly realistic - many neuroimaging effects are weak
- Example: Healthy aging, early-stage neurodegeneration
- Effect size d=0.05-0.12 is typical for subtle differences

**Comparison with sim_datasets:**
- Not explicitly covered in sim_datasets (which uses fixed effect_size=0.15)
- Critical addition for power analysis

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Power analysis | Critical |
| Weak effect detection | Primary goal |
| Clinical relevance | Very high |

**Pros:**
- Tests real-world sensitivity
- Reveals methods' power at clinically relevant effect sizes
- Essential for method selection

**Cons:**
- Results may be unstable (close to noise floor)
- Requires many permutations for reliable p-values
- Ground truth may be partially undetectable by design

**Recommendation:** **Essential benchmark** - sweep effect sizes systematically.

---

### Scenario 5: Signal + Isolated Noise

**Motivation:** Test noise suppression - can methods detect signal while ignoring spurious isolated edges?

**Biological Plausibility:** ★★★★☆
- Realistic - real data has both signal and noise
- Tests specificity alongside sensitivity
- Important for false positive control

**Comparison with sim_datasets:**
- Not covered in sim_datasets
- Unique contribution: explicit noise edges

**Benchmarking Value:**
| Aspect | Assessment |
|--------|------------|
| Specificity testing | Excellent |
| FBC suppression | Core test |
| Real-world relevance | High |

**Pros:**
- Tests both sensitivity AND specificity
- Validates FBC's noise suppression
- Realistic contamination model

**Cons:**
- "Noise" is still structured (1-2 per block)
- Real noise is more heterogeneous
- May be too easy if noise is truly isolated

**Recommendation:** **Keep** - important specificity test.

---

## Part III: Gap Analysis

### Scenarios in sim_datasets.py NOT in fbc_superiority_tests.py

| Scenario | Gap Assessment |
|----------|----------------|
| S1 Single edge | Not needed for FBC (theoretical worst case) |
| S3 Mixed sign | **Missing** - should add |
| S4/S5 Tree/Cycle | Covered implicitly by topological variety |
| S6 Long-range | **Missing** - should add if spatial priors used |
| S7 Multi-component | Partially covered by Scenario 5 |
| S8 Prior aligned | **Relevant** if testing NI-TFNBS |

### Scenarios in fbc_superiority_tests.py NOT in sim_datasets.py

| Scenario | Gap Assessment |
|----------|----------------|
| Scattered stars | **Add to sim_datasets** - unique multi-hub test |
| Checkerboard | Theoretical only - not needed |
| Cross-module pathway | **Add to sim_datasets** - essential |
| Very weak effect | **Add to sim_datasets** - power sweep |
| Signal + noise | **Add to sim_datasets** - specificity test |

---

## Part IV: Recommendations for Benchmarking

### Tier 1: Essential Benchmarks (Must Include)

1. **Within-module dense** (S2) - Core block effect
2. **Between-module pathway** (Scenario 3) - Inter-network effects
3. **Very weak diffuse** (Scenario 4) - Power analysis
4. **Scattered stars** (Scenario 1) - Multi-hub, topologically disconnected

### Tier 2: Important Additions

5. **Long-range shortcuts** (S6) - Spatial structure
6. **Signal + noise** (Scenario 5) - Specificity
7. **Multi-component** (S7) - FWER control
8. **Prior aligned/misaligned** (S8) - For NI-TFNBS

### Tier 3: Stress Tests (Include with Caution)

9. **Mixed sign** (S3) - Sign heterogeneity
10. **Tree/Cycle** (S4/S5) - Topological extremes
11. **Single edge** (S1) - Negative control
12. **Checkerboard** (Scenario 2) - Theoretical extreme

### Effect Size Sweep Recommendation

For realistic benchmarking, sweep across:
- d = 0.05 (very weak - below typical detection)
- d = 0.08 (weak - edge of detection)
- d = 0.12 (moderate-weak - typical neuroimaging)
- d = 0.15 (moderate - optimistic)
- d = 0.20 (strong - large effect)

---

## Part V: Critical Assessment Summary

### sim_datasets.py Strengths
- Comprehensive topology coverage
- Multiple null model families
- Prior-informed scenarios
- Well-structured API

### sim_datasets.py Weaknesses
- Missing between-module pathway
- Fixed effect size (no power analysis)
- No explicit noise/specificity tests
- Some scenarios too artificial (S1)

### fbc_superiority_tests.py Strengths
- Targeted FBC advantage scenarios
- Effect size sweeps
- Noise suppression test
- Biologically motivated multi-hub scenario

### fbc_superiority_tests.py Weaknesses
- Too focused on FBC advantages
- Missing spatial/distance scenarios
- No prior-informed tests
- Some scenarios too artificial (checkerboard)

---

## Conclusion

**Neither framework alone is sufficient for comprehensive benchmarking.**

Recommended synthesis:
1. Use sim_datasets.py as base framework
2. Add fbc_superiority_tests.py scenarios: scattered stars, cross-module, weak effect sweep
3. Always report results across multiple effect sizes
4. Include both sensitivity (power) and specificity (false positives)
5. Acknowledge limitations of synthetic data for all results
