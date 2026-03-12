# AutoResearch Pattern Evaluation for FrontPulse Model Optimization

## Summary

This document evaluates the applicability of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) autonomous experiment loop for optimizing FrontPulse, a time series anomaly detection system focused on onset detection. Three optimization candidates are assessed for feasibility, expected impact, and implementation complexity.

## The AutoResearch Pattern

AutoResearch enables autonomous ML research through a fixed-budget experiment loop:

1. Agent reads `program.md` (human-written research strategy)
2. Agent modifies `train.py` (single editable file containing model + training loop)
3. Training runs for exactly 5 minutes wall-clock time
4. Agent evaluates validation metric (val_bpb for LLMs)
5. Agent decides: keep changes or revert
6. Repeat (~12 experiments/hour, ~100 overnight)

**Key design constraints:**
- Single file modification keeps diffs reviewable
- Fixed time budget ensures fair comparison across architectural changes
- One GPU, one metric, one editable file
- `program.md` is the human-iterable research strategy document

## FrontPulse Context

FrontPulse is a time series anomaly detection system for onset detection (detecting the beginning of anomalies). Target hardware:

- **Training:** RTX 3080 Ti (16 GB VRAM), Ryzen 9 (16 cores), 64 GB RAM
- **Deployment (NAS):** Synology DS224+ (Celeron J4125, 10 GB RAM, no GPU/AVX)

**Evaluation metrics:**
- **NAB (Numenta Anomaly Benchmark):** Rewards early detection, penalizes false positives via sigmoidal scoring within anomaly windows
- **MSD (Mean Signed Difference):** Prediction error with directionality -- critical for onset detection where timing matters
- **EDD (Expected Detection Delay):** Average latency between true anomaly onset and detection
- **ARL (Average Run Length):** Average time until false alarm -- measures robustness against false positives

## Candidate Evaluations

### Candidate 1: Automated MSD Hyperparameter/Feature Subset Search

**Concept:** Use AutoResearch loop to search over hyperparameters and feature subsets, evaluating each configuration against NAB/EDD/ARL on benchmark data.

**Adaptation from AutoResearch pattern:**

| AutoResearch | FrontPulse Adaptation |
|---|---|
| `train.py` (GPT model) | `train.py` (onset detector model + feature pipeline) |
| val_bpb metric | Composite score: weighted NAB + MSD + 1/ARL |
| 5-min fixed budget | 5-min fixed budget (sufficient for tabular/small TS models) |
| `program.md` strategy | `program.md` with search space definition and evaluation protocol |

**Search space:**
- Window sizes, stride, overlap
- Feature subsets (statistical, spectral, wavelet features)
- Detection thresholds, sensitivity parameters
- Regularization, learning rate schedules

**Feasibility: HIGH**

- Hyperparameter search is the most natural fit for the AutoResearch pattern
- 5-minute budget is generous for training small time series models on RTX 3080 Ti
- NAB benchmark datasets are publicly available and well-standardized
- The agent can explore a large space overnight with clear metric feedback

**Expected impact: HIGH**
- Hyperparameter tuning often yields 10-30% improvement on anomaly detection benchmarks
- Feature subset selection can dramatically reduce model complexity for NAS deployment
- Automated search eliminates manual trial-and-error

**Implementation complexity: LOW**
- Straightforward adaptation of AutoResearch's `train.py` pattern
- Replace val_bpb with composite anomaly detection metric
- Use existing NAB evaluation toolkit
- Estimated setup time: 1-2 days

**Recommendation: PROCEED FIRST**

### Candidate 2: Neural Architecture Search for Onset Detector

**Concept:** Use AutoResearch loop to evolve network architecture (layer types, sizes, connections) for the onset detector.

**Adaptation from AutoResearch pattern:**

| AutoResearch | FrontPulse Adaptation |
|---|---|
| Architecture changes in `train.py` | Architecture changes in `train.py` |
| Single model definition | Search space: Conv1D/LSTM/Transformer blocks, skip connections, pooling |
| Agent modifies architecture | Agent modifies architecture definition + hyperparams |

**Search space:**
- Layer types: Conv1D, LSTM, GRU, Transformer, TCN
- Depth: 2-8 layers
- Width: 16-256 units per layer
- Skip connections, attention mechanisms
- Pooling strategies (global, adaptive, none)
- Activation functions

**Feasibility: MEDIUM**

- NAS is well-studied but requires careful search space design
- 5-minute budget limits model complexity (but this is a feature -- ensures NAS-deployable models)
- Risk: agent may get stuck in local optima without sophisticated exploration strategies
- The fixed-budget constraint naturally enforces deployment-friendly architectures

**Expected impact: MEDIUM-HIGH**
- Architecture choice can be more impactful than hyperparameter tuning
- But onset detection may not benefit as much from exotic architectures as NLP/vision tasks
- Constraint to NAS-deployable models limits the upside

**Implementation complexity: MEDIUM**
- Need to define architecture search space as parameterized `train.py`
- Agent must understand how to compose layers (more complex than tuning numbers)
- `program.md` needs careful crafting to guide exploration strategy
- Estimated setup time: 3-5 days

**Recommendation: PROCEED SECOND (after Candidate 1 establishes baseline)**

### Candidate 3: Domain-Specific Small LLM for Paper Classification

**Concept:** Train a small LLM on LITRIS paper data for classification/quality rating, deployable on Synology NAS without API costs.

**Adaptation from AutoResearch pattern:**

| AutoResearch | FrontPulse Adaptation |
|---|---|
| GPT training on text data | Small LLM fine-tuning on paper abstracts/metadata |
| val_bpb metric | Classification accuracy + quality rating correlation |
| H100 GPU | RTX 3080 Ti (16 GB VRAM) |
| Deploy anywhere | Must deploy on Celeron J4125 (no GPU, no AVX) |

**Deployment constraints (Synology DS224+):**
- Celeron J4125: 4 cores, 2.0 GHz, no AVX-512 (AVX2 only in some variants)
- 10 GB RAM total (shared with NAS OS)
- No GPU acceleration
- Target: <4 GB model size, <8s inference latency per paper

**Model candidates:**
- TinyLlama 1.1B (INT4: ~700 MB) -- may work with llama.cpp
- Phi-2 2.7B (INT4: ~1.7 GB) -- good quality/size tradeoff
- SmolLM 135M/360M -- purpose-built for edge deployment
- DistilBERT/TinyBERT -- classification-specific, very small
- Fine-tuned BERT-tiny (15M params) -- most deployable option

**Feasibility: MEDIUM**

- 5-minute budget is tight for LLM fine-tuning even on RTX 3080 Ti
- The real challenge is deployment: Celeron J4125 inference performance is unknown
- LITRIS has 332 papers -- small training set requires careful augmentation
- AutoResearch pattern fits less naturally here (fine-tuning vs. training from scratch)

**Expected impact: MEDIUM**
- Eliminates API costs for routine classification (~$0.01-0.05 per paper currently)
- At 332 papers and growing slowly, annual API cost savings are modest (<$50/year)
- Value is more about independence from external APIs than cost savings
- Quality may degrade vs. Claude/GPT-4 for nuanced quality assessment

**Implementation complexity: HIGH**
- Need training data pipeline from LITRIS extractions
- Fine-tuning setup for multiple small model candidates
- Quantization and deployment testing on NAS hardware
- Inference server setup (llama.cpp or ONNX Runtime)
- Estimated setup time: 5-10 days

**Recommendation: DEFER**
- Cost savings don't justify complexity at current scale
- Revisit when paper count exceeds 1,000+ or if API costs become significant
- If pursued, start with DistilBERT/TinyBERT for classification (simpler than generative LLM)

## Implementation Plan

### Phase 1: Hyperparameter Search (Candidate 1)

1. **Set up AutoResearch fork** adapted for time series
   - Clone autoresearch, adapt directory structure
   - Replace GPT model with onset detector in `train.py`
   - Define composite evaluation metric (NAB + MSD + ARL)

2. **Prepare data pipeline**
   - Download NAB benchmark datasets
   - Create train/val/test splits with proper temporal ordering
   - Implement evaluation metric computation

3. **Write `program.md`**
   - Define search space boundaries
   - Specify exploration vs. exploitation strategy
   - Set acceptance criteria for improvements

4. **Run overnight experiments**
   - ~100 experiments on RTX 3080 Ti
   - Log all results with full reproducibility
   - Analyze Pareto frontier (accuracy vs. model size)

### Phase 2: Architecture Search (Candidate 2)

1. **Define architecture search space** as parameterized `train.py`
2. **Craft NAS-aware `program.md`** that considers deployment constraints
3. **Run search** using Phase 1 winner as baseline
4. **Validate** top-k architectures on held-out test set

### Phase 3 (Future): Edge LLM (Candidate 3)

- Defer until paper count justifies investment
- Start with classification-only models (not generative)
- Benchmark on actual NAS hardware before committing

## Integration with LITRIS

The AutoResearch experiments can leverage existing LITRIS infrastructure:

- **Paper embeddings** (3,746 chunks, Qwen3-Embedding-8B) for domain context
- **Quality ratings** (1-5 scale) as training signal for Candidate 3
- **Citation graph** for understanding research landscape around anomaly detection
- **MCP tools** (`litris_search`) for literature-informed `program.md` iteration

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 5-min budget too short for meaningful training | Low | High | Time series models are small; test with 1-min runs first |
| Agent gets stuck in local optima | Medium | Medium | Add diversity pressure in `program.md`; periodic random restarts |
| NAB metrics don't transfer to production data | Medium | High | Validate on real-world data alongside benchmark |
| NAS hardware too constrained for any model | Low | High (Cand. 3) | Benchmark inference on actual DS224+ before training |
| Small training set limits fine-tuning quality | High (Cand. 3) | Medium | Use augmentation; consider few-shot approaches instead |

## Conclusion

The AutoResearch pattern is a strong fit for FrontPulse optimization, particularly for hyperparameter search (Candidate 1) which maps almost directly onto the existing framework. Architecture search (Candidate 2) is a natural second step. The domain-specific LLM (Candidate 3) should be deferred until scale justifies the complexity.

**Recommended execution order:** Candidate 1 -> Candidate 2 -> Candidate 3 (if warranted)

**Estimated total effort:**
- Candidate 1: 1-2 days setup + overnight runs
- Candidate 2: 3-5 days setup + overnight runs
- Candidate 3: Deferred (5-10 days when pursued)
