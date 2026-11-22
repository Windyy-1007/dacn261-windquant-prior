# Experimental Requirements Specification
## Low-Bit Quantization of Large Language Models: Methods, Sensitivity, and Empirical Analysis

**Document type:** Research implementation specification  
**Project:** HCMUT Specialized Project (DACN)  
**Research scope:** Weight-only post-training quantization, quantization sensitivity, and empirical analysis  
**Primary model:** `meta-llama/Llama-3.2-1B`  
**Secondary model:** `meta-llama/Llama-3.2-3B`

---

# 1. Research Objectives

Implement a reproducible experimental framework for studying low-bit quantization of large language models.

The objective is to investigate:

**RQ1 — Quantization Precision and Model Quality**

How does reducing weight precision affect language-model quality, and how does the degradation differ across model sizes?

**RQ2 — Non-Uniform Quantization Sensitivity**

How does quantization sensitivity vary across Transformer layers, projection types, and fine-grained weight regions?

**RQ3 — Local Error and Global Model Quality**

To what extent do local weight-reconstruction and output-reconstruction errors predict the degradation of the complete language model?

**RQ4 — Interactions and Precision Allocation (Optional)**

Are independently measured sensitivity scores sufficient to explain joint quantization effects, and can these scores support effective mixed-precision allocation under a fixed bit budget?

This project focuses on experimental analysis.

Do not implement or train an RL-based allocator, PPO controller, or WINDQuant-specific optimization procedure.

---

# 2. Global Experimental Protocol

## 2.1 Models

Required:

- `meta-llama/Llama-3.2-1B`

Recommended:

- `meta-llama/Llama-3.2-3B`

Use pretrained base models rather than instruction-tuned variants unless explicitly configured otherwise.

Record:

- Model repository and exact revision
- Parameter count
- Number of Transformer layers
- Number and dimensions of quantizable projection matrices
- Tokenizer revision
- Source precision
- Model configuration

The primary experiment must run successfully on the 1B model before extending to 3B.

## 2.2 Quantization Scope

Study **weight-only quantization**.

Quantizable modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

Unless explicitly configured otherwise:

- Keep token embeddings in original precision.
- Keep the language-model output head in original precision.
- Keep normalization layers in original precision.
- Do not quantize activations or KV cache.
- Do not update pretrained LLM weights through training.

Discover quantizable modules from the model architecture rather than hardcoding their layer counts.

## 2.3 Quantization Backend

Implement one deterministic reference quantizer shared by E1–E6.

Default:

| Parameter | Value |
|---|---|
| Method | Group-wise Round-to-Nearest (RTN) |
| Quantization | Uniform affine |
| Weight bits | 2, 3, 4, 8 |
| Group size | 128 |
| Grouping axis | Contiguous input features per output row |
| Scale/zero-point | Per group |
| Calibration | Not required for basic RTN |
| Stochastic rounding | Disabled |
| Fine-grained analysis chunk | 256 input columns |

The implementation must explicitly document the quantization equations, integer range, scale computation, zero-point convention, clipping and dequantization.

All experiments must use the same quantization operator for a given bit-width.

The chunk size used for sensitivity analysis is conceptually separate from the quantization group size.

Do not silently switch quantization operators, change group size, apply outlier protection, or introduce fallback to higher precision.

A separate advanced backend may be added later, but it must not replace the reference RTN results.

## 2.4 Datasets

Use three logically separated data partitions.

**Calibration dataset**

Default: C4 training split.

- Fixed random seed: 42
- Default sample count: 128
- Default maximum sample length: 512 tokens
- Used for activation statistics and output-reconstruction measurements
- Never use held-out test examples for calibration

**Validation dataset**

Default: WikiText-2 validation split.

- Used for sensitivity measurement and experimental configuration selection
- Use a fixed token subset for the repeated perturbation experiments
- Suggested initial budget: 32,768–65,536 evaluated tokens
- Record the exact evaluated token count

**Test dataset**

Default: WikiText-2 test split.

- Reserved for final evaluation
- Use the full evaluation split where feasible
- Do not optimize bit assignment or choose hyperparameters using test results

Optional external validation:

- C4 validation split

Deduplicate or otherwise check for unintended overlap where datasets or derived samples might overlap.

## 2.5 Evaluation Metrics

Required:

1. Mean negative log-likelihood (NLL)
2. Perplexity (PPL)
3. Absolute NLL degradation relative to full precision
4. Normalized weight reconstruction MSE
5. Quantization payload bits per weight
6. Quantizable parameter count
7. Experiment runtime

For applicable experiments, additionally compute:

- Output reconstruction error
- Effective bit-width for the quantizable subset
- Rank correlation
- Peak GPU memory
- CPU memory usage
- Uncertainty estimates

Evaluate NLL consistently using next-token prediction with correct attention masks, loss masking, context handling and token weighting.

Define:

\[
\mathrm{PPL}=\exp(\mathrm{NLL})
\]

where NLL is the mean token-level cross-entropy over evaluated target tokens.

For comparisons, use NLL as the primary sensitivity metric because it is additive at the token-loss level. PPL is a secondary presentation metric.

## 2.6 Reproducibility Requirements

All experiment results must record:

- Experiment ID
- Model name and revision
- Quantization backend/version
- Bit-width
- Group size
- Dataset and split
- Evaluated token count
- Calibration sample IDs
- Random seed
- Software and GPU information
- Configuration
- Quantization targets
- Runtime
- Metrics
- Git commit, when available

Use YAML or JSON configuration files.

Store results as structured CSV/JSON/JSONL rather than manually copying terminal output.

---

# 3. E1 — Uniform Bit-Width Sweep

**Priority:** P0 — Required

## 3.1 Objective

Measure how model quality changes as weight precision decreases.

Evaluate the relationship among bit-width, reconstruction error and language-model quality.

## 3.2 Experimental Conditions

Models:

- LLaMA-3.2-1B — required
- LLaMA-3.2-3B — recommended

Precision configurations:

- Original FP16/BF16 reference
- W8
- W4
- W3
- W2

Use the same reference RTN backend, group size and quantization scope in all runs.

No mixed-precision allocation is allowed in E1.

## 3.3 Procedure

For each model:

1. Load the original model.
2. Evaluate the full-precision reference on validation data.
3. Save reference metrics.
4. For each bit-width:
   - Restore the original weights.
   - Quantize all selected projection matrices.
   - Evaluate validation NLL and PPL.
   - Measure weight reconstruction error.
   - Calculate nominal payload and effective bit accounting for the defined parameter scope.
   - Evaluate the selected final configurations on test data.
   - Save metrics and configuration.
5. Verify that the reference weights remain unchanged across conditions.

Do not calculate storage savings from fake-quantized tensor sizes. Distinguish theoretical code payload from actual serialized file size.

## 3.4 Required Outputs

**Table E1-A — Quantization Quality**

Columns:

`model, precision, group_size, validation_nll, validation_ppl, test_nll, test_ppl, delta_nll, normalized_weight_mse, quantizable_params, payload_bpw`

**Figure E1-A:** Bit-width versus perplexity.

**Figure E1-B:** Bit-width versus normalized weight reconstruction error.

**Figure E1-C:** Relative NLL degradation across model sizes, if both models are evaluated.

## 3.5 Research Questions

Analyze:

- Is quality degradation monotonic with decreasing precision?
- Which transition causes the largest degradation?
- Are the 1B and 3B models affected differently?
- Does weight reconstruction error track model-quality degradation?
- Does the 2-bit configuration exhibit unstable or catastrophic degradation?

Do not assume the expected trends before observing the results.

## 3.6 Acceptance Criteria

- All five precision conditions are evaluated on 1B.
- All conditions use the same quantization backend.
- Full-precision baseline is reproducible.
- NLL/PPL and reconstruction metrics are recorded.
- At least two publication-quality plots are generated.
- The experiment can be rerun using a saved configuration.

---

# 4. E2 — Layer and Matrix Quantization Sensitivity

**Priority:** P0 — Required

## 4.1 Objective

Determine how sensitive different Transformer layers and projection matrices are to quantization.

The experiment must isolate the effect of quantizing one projection matrix at a time.

## 4.2 Sensitivity Definition

Let \(M\) be the original model.

Let \(M_{i,b}\) denote the model where only matrix \(i\) is quantized to \(b\) bits.

Define:

\[
S_i(b)=
\mathcal L(M_{i,b};D_{\mathrm{val}})
-\mathcal L(M;D_{\mathrm{val}})
\]

where \(\mathcal L\) is mean token-level NLL.

Primary sensitivity measure:

- Delta validation NLL

Secondary measures:

- Relative weight reconstruction MSE
- Output reconstruction error
- Optional delta PPL

## 4.3 Experimental Conditions

Primary model:

- LLaMA-3.2-1B

Quantization levels:

- W2 — required
- W3 — recommended
- W4 — required if E6 will be implemented

Quantize all supported attention and MLP projection matrices individually.

Each trial must modify exactly one matrix.

Do not quantize the entire Transformer block when measuring a single matrix.

## 4.4 Procedure

1. Load the original model.
2. Record the reference validation NLL.
3. Enumerate the target projection matrices.
4. For each target matrix:
   - Preserve its original tensor.
   - Apply the specified RTN quantization to that matrix alone.
   - Evaluate validation NLL.
   - Compute delta NLL.
   - Compute local reconstruction error.
   - Restore the original tensor.
   - Verify restoration.
5. Repeat for each configured bit-width.
6. Aggregate results by Transformer depth and projection type.

The implementation must ensure that previously quantized matrices do not remain modified when evaluating the next matrix.

Prefer temporary in-place replacement with deterministic restoration over repeatedly loading the full model.

## 4.5 Required Outputs

**Table E2-A — Matrix Sensitivity**

Columns:

`model, layer_idx, module_name, projection_type, bit_width, n_parameters, baseline_nll, quantized_nll, delta_nll, normalized_weight_mse, output_mse`

**Figure E2-A:** Heatmap of layer index versus projection type, colored by delta NLL.

**Figure E2-B:** Distribution of sensitivity across projection types.

**Figure E2-C:** Sensitivity versus Transformer depth.

**Table E2-B:** Most sensitive and least sensitive matrices.

## 4.6 Research Questions

- Are all layers equally sensitive?
- Are attention projections more or less sensitive than MLP projections?
- Is sensitivity associated with layer depth?
- Which matrices exhibit unusually large quality degradation?
- Do matrix rankings change between W2, W3 and W4?

## 4.7 Acceptance Criteria

- All target projection matrices are tested at W2.
- Each sensitivity score comes from a single-matrix perturbation.
- No cross-trial quantization contamination is permitted.
- Heatmaps and aggregate distributions are generated.
- The original model's outputs are recovered after restoration within numerical tolerance.

---

# 5. E3 — Fine-Grained Column-Chunk Sensitivity

**Priority:** P0 — Required

## 5.1 Objective

Investigate whether quantization sensitivity varies between subregions of the same weight matrix.

This experiment must measure sensitivity directly rather than infer sensitivity from a learned bit-allocation policy.

## 5.2 Experimental Conditions

Primary model:

- LLaMA-3.2-1B

Primary precision:

- W2

Optional:

- W3

Allocation-analysis chunk size:

- 256 consecutive input columns

Reference quantizer group size:

- 128 input features

Use a predetermined set of representative matrices.

Suggested selection:

- One early Transformer layer
- One middle Transformer layer
- One late Transformer layer

For each selected layer:

- `q_proj`
- `gate_proj`
- `down_proj`

Total:

- 9 representative matrices, where these modules exist

Choose layer indices from the model configuration rather than assuming a specific number of layers.

Record the selection rule before inspecting results to avoid cherry-picking.

## 5.3 Sensitivity Definition

For a weight matrix \(W\), partition its input dimension into column chunks:

\[
U_j=W[:,c_j:c_j+G]
\]

For each chunk:

\[
S_j(b)=
\mathcal L(M_{j,b};D_{\mathrm{val}})
-\mathcal L(M;D_{\mathrm{val}})
\]

Only the selected chunk is quantized.

All other weights remain at original precision.

## 5.4 Procedure

1. Select representative matrices.
2. Partition each matrix into contiguous chunks.
3. For each chunk:
   - Preserve original values.
   - Quantize only the selected chunk.
   - Evaluate validation NLL.
   - Calculate delta NLL.
   - Record chunk coordinates, size and reconstruction error.
   - Restore original values.
4. Aggregate sensitivity measurements within each matrix.
5. Compare chunk-level distributions.

Keep quantization group boundaries consistent. The chosen chunk size must not accidentally alter the RTN quantization rule.

## 5.5 Required Outputs

**Table E3-A — Chunk Sensitivity**

Columns:

`model, layer_idx, module_name, chunk_idx, col_start, col_end, bit_width, n_parameters, delta_nll, normalized_weight_mse, output_mse`

**Figure E3-A:** Chunk index versus delta NLL for representative matrices.

**Figure E3-B:** Within-matrix sensitivity distribution.

**Figure E3-C:** Normalized chunk sensitivity heatmap.

**Table E3-B:** Per-matrix sensitivity statistics:

- Mean
- Standard deviation
- Median
- Minimum
- Maximum
- Interquartile range
- Range or spread

Use robust statistics where scores are close to zero. Avoid unstable ratios with near-zero denominators.

## 5.6 Research Questions

- Does sensitivity vary within the same projection matrix?
- How large is the within-matrix variation?
- Do sensitivity patterns differ between attention and MLP matrices?
- Are locally high-error chunks necessarily globally sensitive?
- Does the selected 256-column analysis granularity reveal meaningful heterogeneity?

## 5.7 Acceptance Criteria

- At least 6 representative matrices are evaluated.
- All available column chunks in each selected matrix are tested at W2.
- The quantization backend remains unchanged.
- A per-chunk sensitivity dataset is generated.
- Within-matrix heterogeneity is reported quantitatively.

---

# 6. E4 — Local Reconstruction Error vs Global Model Quality

**Priority:** P0 — Required

## 6.1 Objective

Determine whether local quantization errors are reliable proxies for global language-model degradation.

This experiment reuses the measurements generated in E2 and E3 wherever possible.

## 6.2 Metrics

For each quantized unit \(i\):

**Weight reconstruction error**

\[
E_i^W =
\frac{\|W_i-\widehat W_i\|_F^2}
{\|W_i\|_F^2+\epsilon}
\]

**Output reconstruction error**

For the associated input activation \(X_i\):

\[
E_i^Y =
\frac{\|X_iW_i^\top-X_i\widehat W_i^\top\|_F^2}
{\|X_iW_i^\top\|_F^2+\epsilon}
\]

Use representative cached activations collected from calibration data.

For chunk-level experiments, use the corresponding input features and weight slice so the local contribution is dimensionally correct.

**Global quality degradation**

\[
S_i=
\mathcal L(M_i;D_{\mathrm{val}})
-\mathcal L(M;D_{\mathrm{val}})
\]

## 6.3 Procedure

1. Load E2 and E3 result datasets.
2. Ensure that reconstruction metrics have been computed using consistent definitions.
3. Join measurements by model, layer, module, region and bit-width.
4. Calculate Spearman rank correlation between:
   - Weight MSE and delta NLL
   - Output reconstruction error and delta NLL
5. Optionally compute Pearson correlation.
6. Repeat the analysis separately for:
   - Matrix-level measurements
   - Chunk-level measurements
   - Attention projections
   - MLP projections
7. Estimate uncertainty for correlations using a suitable bootstrap scheme that respects dependence between samples or modules.
8. Report ranking disagreements and outliers.

Do not pool normalized and unnormalized measurements without explicitly accounting for differences in tensor size.

## 6.4 Required Outputs

**Table E4-A — Correlation Analysis**

Columns:

`granularity, module_group, bit_width, n_units, spearman_weight_vs_nll, spearman_output_vs_nll, confidence_interval`

**Figure E4-A:** Weight reconstruction error versus delta NLL.

**Figure E4-B:** Output reconstruction error versus delta NLL.

**Figure E4-C:** Ranking agreement between local proxies and measured sensitivity.

## 6.5 Research Questions

- Does low weight reconstruction error imply low model degradation?
- Is activation/output-aware error more predictive than weight MSE?
- Do relationships differ between attention and MLP?
- Are observed correlations stable across precision levels and unit granularities?

## 6.6 Acceptance Criteria

- Both local error measures are available.
- Global quality degradation is taken from E2/E3, not inferred from reconstruction MSE.
- Correlation analysis is performed at matrix and chunk level.
- Results include sample counts and uncertainty information.
- The analysis does not assume correlations must be weak.

---

# 7. E5 — Cross-Region Quantization Error Interaction

**Priority:** P1 — Recommended

## 7.1 Objective

Test whether the combined effect of quantizing multiple weight regions is adequately approximated by the sum of their individual effects.

## 7.2 Interaction Definition

For two regions \(i,j\), define:

\[
I_{ij}=
\Delta\mathcal L(\{i,j\})
-\Delta\mathcal L(\{i\})
-\Delta\mathcal L(\{j\})
\]

where:

\[
\Delta\mathcal L(A)=
\mathcal L(M_A;D_{\mathrm{val}})
-\mathcal L(M;D_{\mathrm{val}})
\]

Positive interaction indicates the combined degradation is greater than the sum of separate degradations.

Negative interaction indicates the combined degradation is smaller.

This is a finite-difference interaction measure, not a causal attribution to an internal Transformer mechanism.

## 7.3 Experimental Conditions

Primary model:

- LLaMA-3.2-1B

Precision:

- W2

Evaluate at least 20 distinct pairs if computationally feasible.

Include:

- Pairs within the same matrix
- Pairs in different matrices of the same layer
- Pairs in different Transformer layers
- Pairs with high/high, high/low and low/low individually measured sensitivities

Record how pairs were selected.

Where feasible, include a fixed-seed random sample to reduce selection bias.

## 7.4 Procedure

For each pair:

1. Evaluate or load reference NLL.
2. Load individual sensitivity scores from E2/E3.
3. Restore original weights.
4. Quantize both selected regions simultaneously.
5. Evaluate validation NLL.
6. Calculate joint degradation.
7. Calculate the additive prediction.
8. Calculate interaction residual.
9. Restore both regions.
10. Save results.

Use fixed quantized weight values for each region across individual and joint experiments. Do not introduce different quantizer fits merely because another region was perturbed.

## 7.5 Required Outputs

**Table E5-A — Pairwise Interaction**

Columns:

`region_i, region_j, pair_type, bit_width, delta_i, delta_j, predicted_additive_delta, observed_joint_delta, interaction_residual`

**Figure E5-A:** Predicted versus observed joint degradation.

**Figure E5-B:** Interaction residuals by pair category.

**Figure E5-C:** Distribution of positive and negative interactions.

## 7.6 Research Questions

- Are quantization effects approximately additive?
- How large are interaction residuals relative to measurement uncertainty?
- Do interactions differ between same-layer and cross-layer pairs?
- Can an independently measured sensitivity model adequately predict combined degradation?

## 7.7 Acceptance Criteria

- At least 20 pair experiments are completed, or limitations are documented.
- Individual and joint perturbations use matching quantization parameters.
- Both positive and negative interaction outcomes are reported without filtering.
- Uncertainty and evaluation noise are discussed.
- Conclusions are restricted to the sampled pairs.

---

# 8. E6 — Uniform vs Sensitivity-Based Mixed Precision

**Priority:** P2 — Optional Extension

## 8.1 Objective

Determine whether non-uniform bit allocation based on measured sensitivity improves quality over simple allocation strategies under matched precision budgets.

This is an empirical baseline study, not a new RL-based optimization method.

## 8.2 Allocation Strategies

Implement:

**A. Uniform precision**

Assign the same bit-width to all quantizable matrices.

**B. Random mixed precision**

Assign bit-widths from the configured action set while satisfying a target budget.

Use multiple seeds.

**C. Sensitivity-based greedy allocation**

Use E2 validation sensitivity estimates to rank precision upgrades by expected reduction in NLL per additional bit.

Possible action set:

\[
\mathcal A=\{2,3,4\}
\]

Do not include an unrelated quantization backend.

## 8.3 Optimization Formulation

Given sensitivity estimates \(d_i(b)\), solve approximately:

\[
\min_{\mathbf b}
\sum_i d_i(b_i)
\]

subject to:

\[
\sum_i n_i b_i\le B
\]

where \(n_i\) is the parameter count of matrix \(i\).

This is a surrogate objective using independently measured sensitivity; it is not the actual jointly measured model loss.

The final quality must therefore be evaluated by quantizing the entire model according to the selected plan.

## 8.4 Experimental Conditions

Primary model:

- LLaMA-3.2-1B

Optional:

- LLaMA-3.2-3B

Target average precision:

- 2.5 bits
- 3.0 bits
- 3.5 bits

Use an explicitly defined quantizable-parameter denominator.

If exact targets are not achievable due to discrete matrix sizes, report the realized precision and budget deviation.

Do not claim a matched-budget comparison when the realized bit-widths differ materially.

At integer bit budgets, include the corresponding uniform baseline. At fractional budgets, random and sensitivity-based mixed-precision strategies should be compared at matched realized costs, while uniform W2/W3/W4 may be presented as reference endpoints rather than exact-budget competitors.

## 8.5 Procedure

1. Load sensitivity data from E2 for each candidate bit-width.
2. Compute the effective cost of assigning each matrix a candidate precision.
3. Start with the lowest allowed precision.
4. Greedily select upgrades according to predicted quality improvement per additional bit.
5. Continue until the budget limit is reached or no beneficial feasible upgrade remains.
6. Construct the final quantized model.
7. Evaluate validation and held-out test quality.
8. Generate random mixed-precision allocations with comparable realized cost.
9. Compare quality against uniform reference points.
10. Save each bit-allocation plan.

The test split must not be used to select the bit allocation.

## 8.6 Required Outputs

**Table E6-A — Precision Allocation Comparison**

Columns:

`strategy, target_bpw, realized_bpw, validation_nll, test_nll, test_ppl, delta_nll, n_matrices_w2, n_matrices_w3, n_matrices_w4, runtime`

**Figure E6-A:** NLL versus effective bits.

**Figure E6-B:** Per-matrix precision allocation heatmap.

**Figure E6-C:** Quality comparison at matched effective bit budgets.

## 8.7 Research Questions

- Does sensitivity-based allocation improve over random mixed precision?
- Does mixed-precision allocation improve over uniform precision when costs are comparable?
- Do sensitivity scores measured independently remain predictive when many matrices are quantized simultaneously?
- How does allocation performance depend on the global bit budget?

## 8.8 Acceptance Criteria

- All strategies use the same backend and parameter scope.
- Realized bit budgets are explicitly reported.
- Comparisons at matched costs are clearly distinguished from reference endpoints.
- The final model is evaluated as a complete quantized model.
- At least one quality–precision trade-off curve is generated.

---

# 9. Shared Implementation Requirements

## 9.1 Code Organization

Suggested directory structure:

```text
dacn_quantization_study/
├── configs/
│   ├── common.yaml
│   ├── e1.yaml
│   ├── e2.yaml
│   ├── e3.yaml
│   ├── e4.yaml
│   ├── e5.yaml
│   └── e6.yaml
├── src/
│   ├── models/
│   ├── datasets/
│   ├── quantization/
│   ├── evaluation/
│   ├── sensitivity/
│   ├── allocation/
│   ├── metrics/
│   └── utils/
├── experiments/
│   ├── e1_uniform_sweep.py
│   ├── e2_matrix_sensitivity.py
│   ├── e3_chunk_sensitivity.py
│   ├── e4_error_correlation.py
│   ├── e5_pairwise_interactions.py
│   └── e6_mixed_precision.py
├── tests/
├── results/
├── figures/
└── README.md
```

Existing quantization utilities may be reused when they satisfy the reference backend specification.

Do not restructure an existing repository unnecessarily.

Do not modify unrelated research code or existing experiment results.

## 9.2 Restoration Correctness

Single-region perturbation is central to E2, E3 and E5.

Implement a safe mechanism to:

- Save the original tensor or tensor slice.
- Quantize the selected region.
- Evaluate the perturbed model.
- Restore the original data.
- Verify restoration.

Restoration must also occur when evaluation raises an exception.

Include automated tests ensuring no changes remain outside the selected region.

For the small-scale validation setup, compare restored tensors exactly where possible and compare reference logits within numerical tolerance.

## 9.3 Activation Collection

For E4:

- Register forward hooks only on required modules.
- Capture representative calibration inputs.
- Limit cached tensors to a configurable memory budget.
- Store or stream activation statistics safely.
- Remove hooks after collection.

Do not retain full activations for all layers and all tokens if this causes unnecessary memory usage.

## 9.4 Evaluation Efficiency

The pipeline must support:

- Reusing the loaded reference model
- Reusing tokenized datasets
- Reusing reference NLL
- Reusing quantized weight candidates when valid
- Incremental result writing
- Resuming incomplete experiments
- Skipping completed configurations

An interrupted experiment must not force all completed perturbations to be reevaluated.

## 9.5 Statistical Reporting

Report exact sample counts.

Where meaningful:

- Use paired comparison on the same validation tokens.
- Estimate uncertainty using sequence-level or document-level resampling.
- Report confidence intervals for aggregate sensitivity or correlations.
- Avoid treating highly dependent chunks from one matrix as fully independent samples.
- Avoid claiming statistical significance without an appropriate test.

## 9.6 Visualization

All figures must be publication-ready:

- Clear axis labels and units
- Readable typography
- Legends for multiple methods
- Explicit precision labels
- Correct NLL/PPL distinction
- No truncated axes without disclosure
- Suitable for embedding in a LaTeX report

Export each figure in:

- PDF
- PNG

Also export the underlying data as CSV.

Do not generate illustrative or fabricated experiment values in final result files.

---

# 10. Deliverables

The completed experimental framework must provide:

**Code**

- Reproducible experiment scripts
- Configuration files
- Quantization and perturbation utilities
- Evaluation utilities
- Automated unit and integration tests

**Results**

- Per-experiment CSV/JSON results
- Aggregated summary tables
- Hardware/software metadata
- Run configurations
- Experimental logs

**Figures**

- E1: Precision–quality curves
- E2: Layer/module sensitivity heatmap
- E3: Within-matrix chunk sensitivity analysis
- E4: Reconstruction error and global quality correlations
- E5: Pairwise interaction analysis, if enabled
- E6: Mixed-precision trade-off curves, if enabled

**Documentation**

- README with setup and run instructions
- Quantization mathematical definition
- Dataset and evaluation protocol
- Experiment-specific descriptions
- Known limitations
- Reproducibility checklist

**Research Summary**

Create a Markdown report containing:

1. Experimental objective
2. Method
3. Dataset/model configuration
4. Result tables
5. Figures
6. Observations
7. Limitations
8. Interpretation relevant to the DACN research questions

Avoid presenting expected findings as measured results.

---

# 11. Execution Plan

Implement and execute in this order.

## Phase A — Framework Validation

- Validate model loading and evaluation.
- Implement deterministic RTN.
- Test quantize/dequantize correctness.
- Validate full-precision perplexity calculation.
- Implement safe tensor restoration.
- Verify configuration logging and result persistence.

**Gate A:** Quantization is deterministic, evaluation works and no model-state contamination occurs.

## Phase B — Core Experiments

Run:

1. E1 — Uniform Bit-Width Sweep
2. E2 — Layer/Matrix Sensitivity
3. E3 — Fine-Grained Chunk Sensitivity
4. E4 — Error Correlation

**Gate B:** All P0 result tables and figures are generated from genuine measurements.

## Phase C — Extended Analysis

Run if resources permit:

5. E5 — Pairwise Quantization Interactions
6. E6 — Sensitivity-Based Mixed Precision

**Gate C:** At least one additional question about independent sensitivity or budgeted allocation is empirically investigated.

## Phase D — Research Reporting

- Aggregate results.
- Generate final figures.
- Write methodological descriptions.
- Record limitations.
- Prepare experiment tables for Chapter 5 of the DACN report.

---

# 12. Research Integrity Constraints

The implementation must satisfy the following constraints:

1. Do not fabricate metrics, plots or experimental results.
2. Do not infer matrix sensitivity from a learned allocator's bit assignments.
3. Do not confuse local reconstruction error with global model-quality degradation.
4. Do not use the final test split for allocation optimization.
5. Do not claim matching bit budgets unless realized costs are comparable.
6. Do not claim runtime memory savings from fake quantization.
7. Do not claim an observed correlation proves causation.
8. Do not interpret one-model results as universal across LLM architectures.
9. Do not report results as reproduced unless they were actually rerun.
10. Keep published baseline results separate from results generated by this experimental framework.

---

# 13. Final Definition of Done

The required DACN experimental work is complete when:

- E1–E4 pass their acceptance criteria.
- All results are reproducible from saved configurations.
- Source code and experiment artifacts are available.
- At least one uniform bit-width analysis is completed.
- At least one complete matrix-level sensitivity map is generated.
- At least one within-matrix sensitivity analysis is generated.
- The relationship between local reconstruction error and global NLL degradation is quantified.
- Findings and limitations are written without assuming a particular desired conclusion.
- The resulting tables and plots can directly support Chapter 5, *Experiments and Evaluation*.

E5 and E6 are optional and must not block completion of the required four experiments.

**End of specification.**