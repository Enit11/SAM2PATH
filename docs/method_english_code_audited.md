# Method

## Code-consistency note before submission

This draft follows the current training path and the formal configuration `configs/Glomeruli_uni2_spherical_hrrefine.py`. Four discrepancies must be resolved or explicitly excluded before submission. First, the SAM prompt encoder is trainable (`freeze.prompt_encoder=False`) and is therefore included in the optimizer, although it was omitted from an earlier Chinese description. Second, the memory-specific configuration fields `min_region_pixels`, `min_region_purity`, and `conf_threshold` are not independently consumed by the current memory implementation; it reuses prototype-derived region and confidence criteria. Third, `broadcast_coef` is present in the configuration but is not read by the training module and has no effect on optimization. Fourth, the current unsupervised geometry term compares the boundary prediction of the student on a labeled image with the boundary prediction of the teacher on an unlabeled weak view. These tensors are not corresponding views of the same image. Accordingly, the term is denoted below as an implementation-dependent auxiliary term and is **not** presented as valid geometry consistency. It should be removed, disabled, or corrected to compare matched weak/strong views before a final paper is prepared.

## 3.1. Problem Formulation and Overall Framework

We consider semi-supervised semantic segmentation of kidney pathology images into background/non-target tissue, glomerulus, renal vessel, and renal tubule. Let

\[
\mathcal{D}_l=\{(x_i,y_i)\}_{i=1}^{N_l},\qquad
\mathcal{D}_u=\{x_j\}_{j=1}^{N_u},
\tag{1}
\]

denote the labeled and unlabeled sets, respectively. Each input image is resized to \(1024\times1024\), and \(y\in\{0,1,2,3\}^{H\times W}\), where labels 0--3 indicate background, glomerulus, vessel, and tubule. The student network \(f_{\theta}\) produces four-channel segmentation logits \(L_s=f_{\theta}(x)\in\mathbb{R}^{4\times H\times W}\), from which pixel probabilities are obtained as \(p_s=\operatorname{softmax}(L_s)\).

Our architecture combines a frozen SAM2 Hiera image encoder with a frozen UNI2-h Transformer. A trainable UNI neck maps UNI features to the same channel dimension as the SAM2 representation. The aligned features are processed by a convolutional fusion path and a spatially gated, unit-hypersphere-normalized residual branch. The resulting feature map is passed through a trainable dual-branch adapter and the SAM mask decoder. In parallel, the highest-resolution SAM2 FPN feature is used to refine the decoder logits. During semi-supervised training, an exponential-moving-average (EMA) teacher processes weakly augmented unlabeled images, whereas the student processes strongly augmented images. Confidence-filtered teacher regions update class prototypes and a class-wise instance memory, which in turn regularize student region features.

The framework comprises two coupled objectives. Region-level semantic regularization addresses unreliable pseudo-labels and cross-class feature ambiguity. Geometry-aware supervision and high-resolution logit refinement address contour degradation caused by coarse decoder features. These branches share the same student representation and final segmentation path, but no explicit boundary-ownership or core-to-boundary assignment mechanism is implemented.

**Architecture figure specification.** Draw two image-encoder streams: frozen SAM2 Hiera and frozen UNI2-h Transformer followed by a trainable UNI neck. Align both outputs at \(64\times64\), concatenate them, and feed them to a 512-to-256 fusion neck. Add a spatially gated normalized residual branch to this output. Feed the fused feature into the adapter and SAM decoder. Connect the largest SAM2 FPN feature to a logit refiner together with the low-resolution decoder logits. Below the student path, draw weak unlabeled images to the EMA teacher, then confidence filtering, connected components, prototype/memory buffers, and arrows back to prototype and memory losses on student features. Draw labeled masks feeding supervised and geometry losses.

## 3.2. Cross-encoder Spatial Adaptation and Unit-Hypersphere Gated Residual Fusion

### Motivation

SAM2 and UNI2-h provide representations with different pretraining objectives and feature statistics. A direct combination of their raw outputs would conflate channel projection, spatial alignment, and semantic fusion. We therefore first adapt the two encoder outputs to a common spatial feature space and then introduce a residual normalized fusion branch.

### Feature adaptation and concatenation path

The frozen SAM2 encoder provides \(F^S\in\mathbb{R}^{B\times256\times64\times64}\). The frozen UNI2-h Transformer is followed by a trainable convolutional neck, producing \(F^U\in\mathbb{R}^{B\times256\times64\times64}\). If their spatial sizes differ, the UNI feature is bilinearly interpolated to the SAM2 grid. The primary fusion path is

\[
F^c=[F^S;F^U]\in\mathbb{R}^{B\times512\times64\times64},\qquad
F^f=N_f(F^c)\in\mathbb{R}^{B\times256\times64\times64},
\tag{2}
\]

where \([\cdot;\cdot]\) denotes channel concatenation and \(N_f\) is the trainable convolutional fusion neck.

### Spatially gated normalized residual branch

To compare encoder features primarily by direction rather than feature norm, we project and normalize each stream along the channel dimension:

\[
Z^S=\operatorname{Norm}(W_S(F^S)),\qquad
Z^U=\operatorname{Norm}(W_U(F^U)),
\tag{3}
\]

where \(\operatorname{Norm}(v)=v/(\lVert v\rVert_2+\epsilon)\), \(\epsilon=10^{-6}\), and \(W_S,W_U\) are trainable \(1\times1\) projections followed by normalization layers. A spatial gate is computed as

\[
G=\sigma\bigl(\mathcal{G}([Z^S;Z^U])\bigr)\in[0,1]^{B\times1\times64\times64}.
\tag{4}
\]

The normalized gated feature and final fused representation are

\[
Z=\operatorname{Norm}\bigl(G\odot Z^S+(1-G)\odot Z^U\bigr),\qquad
F=F^f+\alpha R(Z),
\tag{5}
\]

where \(R\) is a trainable \(1\times1\) residual projection and \(\alpha=0.1\). The convolution in the final residual projection is initialized to zero. Hence, the added branch initially preserves the output of the concatenation path and learns a correction during optimization.

This operation is not cross-attention: it does not form query--key--value interactions or token-pair attention matrices. It is also not a recurrent or streaming feature-fusion process. The term *unit-hypersphere gated residual fusion* is mathematically accurate because the projected features and their gated combination are L2-normalized. It should not be described as a Riemannian, geodesic, or Fréchet-mean fusion, since no logarithmic/exponential map or intrinsic spherical averaging is implemented.

## 3.3. Confidence-filtered Mean Teacher Learning

### Motivation

Unlabeled images provide additional appearance variation, but teacher predictions can be unreliable near tissue interfaces or under weak visual contrast. We use a teacher--student formulation in which confidence filtering determines which teacher predictions may supervise the student and update region-level semantic states.

The teacher parameters are updated after each training batch as

\[
\theta_t^{(k+1)}=\beta\theta_t^{(k)}+(1-\beta)\theta_s^{(k)},
\tag{6}
\]

where \(\beta=0.999\). The teacher is evaluated without gradients on a weakly augmented view \(x_u^w\); the student receives a strongly augmented view \(x_u^s\). Given teacher logits \(L_t\), let

\[
p_t(i)=\operatorname{softmax}(L_t(i)),\qquad q(i)=\max_c p_{t,c}(i).
\tag{7}
\]

The consistency-valid set is \(\Omega_{mt}=\{i:q(i)\geq\tau_{mt}\}\), with \(\tau_{mt}=0.8\). The implemented consistency objective is

\[
\mathcal{L}_{mt}=\frac{1}{|\Omega_{mt}|}\sum_{i\in\Omega_{mt}}\left\|p_s(i)-p_t(i)\right\|_2^2.
\tag{8}
\]

Its coefficient is exponentially warmed up to 1.0 over the first 10 epochs. For prototype and memory operations, we use the stricter foreground criterion

\[
\Omega_p=\{i:q(i)\geq\tau_p\ \wedge\ \arg\max_c p_{t,c}(i)>0\},\qquad \tau_p=0.9.
\tag{9}
\]

Thus, low-confidence and background pseudo-labels do not enter the foreground prototype or memory updates. The current implementation does not use class-specific thresholds, top-1/top-2 probability margins, topology reliability, or a dedicated boundary uncertainty filter. These are limitations rather than implemented components.

## 3.4. Confidence-filtered Region-level Prototype Matching and Instance Memory

### Region construction and class prototypes

Pixel-wise overlap losses do not explicitly require distinct connected structures of the same tissue class to occupy a stable semantic neighborhood. We therefore form 8-connected foreground regions from ground-truth masks or from pseudo-labels retained by Eq. (9). Let \(\mathcal{R}_c\) denote the accepted regions of class \(c\in\{1,2,3\}\). Background is supervised in segmentation but is intentionally excluded from the prototype and memory banks.

For a teacher region \(R\in\mathcal{R}_c\), the normalized region token is

\[
z_R=\operatorname{Norm}\left(\frac{1}{|R|}\sum_{i\in R}F_t(i)\right).
\tag{10}
\]

For each foreground class, a foundation prototype is maintained using a no-gradient momentum update:

\[
P_c\leftarrow\operatorname{Norm}\left(mP_c+(1-m)z_R\right),\qquad m=0.99.
\tag{11}

The first accepted region initializes an unready prototype. Prototype losses are evaluated only when at least two foreground prototypes are ready and after epoch 5.

For a student region \(R\) with label \(y_R\), pixel features are normalized before pooling:

\[
u_R=\operatorname{Norm}\left(\frac{1}{|R|}\sum_{i\in R}\operatorname{Norm}(F_s(i))\right).
\tag{12}
\]

Let \(\mathcal{C}_{ready}\) be the available foreground prototype classes and \(\tau_p=0.1\). The prototype matching loss is

\[
\mathcal{L}_{reg}(R)=-\log\frac{\exp(u_R^\top P_{y_R}/\tau_p)}
{\sum_{c\in\mathcal{C}_{ready}}\exp(u_R^\top P_c/\tau_p)}.
\tag{13}
\]

The intra-region compactness term is

\[
\mathcal{L}_{intra}(R)=\frac{1}{|R|}\sum_{i\in R}\left[1-\operatorname{Norm}(F_s(i))^\top u_R\right].
\tag{14}

and the implemented cross-class prototype separation term is

\[
\mathcal{L}_{sep}(R)=\frac{1}{|\mathcal{C}_{ready}|-1}\sum_{c\neq y_R}
\max\left(0,u_R^\top P_c-\delta\right),\qquad\delta=0.2.
\tag{15}

These terms supplement Dice, cross-entropy, and pixel-wise consistency by comparing connected regions against persistent class-level semantic references. They do not themselves establish instance boundaries or topology.

### Semantic purification as a procedure, not a learnable network

In this work, *semantic purification* denotes the procedure of confidence filtering, foreground connected-component extraction, minimum-area filtering, prototype matching, intra-region compactness, and memory-based regularization. It is not an independent neural network or decoder. The prototype path accepts connected components with at least six pixels, processes at most 16 regions per image, and samples at most 128 pixels per region. The code includes a nominal region-purity calculation; however, because connected components are already extracted from a class-specific binary mask, this quantity is not a strong test of cross-class contamination. We therefore describe it conservatively as an implementation-level region-validity check.

### Class-wise FIFO instance memory

To retain multiple historical instance representations per class, a FIFO memory of 16 slots is maintained for each foreground class. For an accepted teacher token \(z_R\) written into class \(c\), an occupied slot is updated by

\[
M_{c,k}\leftarrow\operatorname{Norm}\left(\rho M_{c,k}+(1-\rho)z_R\right),\qquad\rho=0.9,
\tag{16}

and the class-specific pointer advances cyclically. Memory updates start after epoch 10. In the current code, the memory path uses the prototype-derived confidence mask and minimum region size; the stricter memory-only threshold fields in the configuration are not independently enforced.

For a student instance token \(u_R\), the feature-matching loss is

\[
\mathcal{L}_{mem}^{feat}(R)=1-\max_{k\in\mathcal{K}_{y_R}}u_R^\top M_{y_R,k}.
\tag{17}

Let \(\mathcal{M}\) be all ready memory tokens and \(c_j\) the class of token \(M_j\). The memory contrastive loss is

\[
\mathcal{L}_{mem}^{logit}(R)=-\log
\frac{\sum_{j:c_j=y_R}\exp(u_R^\top M_j/\tau_m)}
{\sum_{j\in\mathcal{M}}\exp(u_R^\top M_j/\tau_m)},\qquad\tau_m=0.1.
\tag{18}

Prototypes represent a single momentum-averaged class center, whereas memory retains several historical instance tokens and may preserve within-class variation. This is a design rationale, not an established benefit; prototype-only, memory-only, and combined ablations are required to determine whether the memory provides non-redundant information.

## 3.5. Geometry-aware Supervision and High-resolution Logit Refinement

### Geometry targets and hard boundary supervision

Region overlap losses can be relatively insensitive to small contour displacements. From labeled masks, we construct a dilated semantic boundary band \(B\) by detecting local label changes and applying morphological dilation. We also construct one truncated signed distance map for each foreground class, \(D\in\mathbb{R}^{3\times H\times W}\). The geometry heads predict a boundary logit \(\hat{b}\) and three distance maps \(\hat{D}\). Their losses are

\[
\mathcal{L}_{b}=\operatorname{WeightedBCEWithLogits}(\hat{b},B),\qquad
\mathcal{L}_{d}=\operatorname{SmoothL1}(\hat{D},D).
\tag{19}
\]

The boundary BCE uses a positive-class weight derived from the batch foreground-boundary ratio and a configured label-dependent spatial weight. **Implementation caution:** the variable named `vessel_boundary_weight` is currently applied to `label==3`, which denotes tubule under the stated label convention. We therefore do not attribute the weighting to vessels in this manuscript draft.

Hard boundary pixels are selected from labeled GT boundary locations by

\[
\Omega_h=\left\{i:B(i)=1\ \wedge\ \left[\mathcal{H}(p_s(i))\geq\eta\ \vee\ L_{(1)}(i)-L_{(2)}(i)\leq\gamma\right]\right\},
\tag{20}

where \(\mathcal{H}\) is entropy, \(L_{(1)}\) and \(L_{(2)}\) are the largest and second-largest class logits, \(\eta=0.45\), and \(\gamma=2.0\). The hard-boundary objective is the mean cross-entropy on \(\Omega_h\). Boundary, distance, and hard-boundary supervision begin after epoch 10.

### High-resolution logit refinement

The decoder produces low-resolution four-class logits \(L_{low}\in\mathbb{R}^{B\times4\times256\times256}\). The model selects the largest-spatial-resolution SAM2 FPN feature \(F_{hr}\) and interpolates it to the spatial size of \(L_{low}\). A trainable refiner predicts a residual correction:

\[
L_{ref}=L_{low}+\alpha_r H_r\left(\operatorname{Up}(F_{hr}),L_{low}\right),\qquad\alpha_r=0.1.
\tag{21}

The final convolution of \(H_r\) is zero-initialized, so the initial refined logits equal the original decoder logits. The refined logits are bilinearly upsampled to \(1024\times1024\). This branch directly modifies the final class competition, unlike the auxiliary geometry heads. Geometry supervision and logit refinement are motivated as complementary, but the code does not feed geometry-head outputs into the refiner; any claim of synergy requires ablation.

## 3.6. Joint Objective and Optimization

For labeled images, the implemented supervised loss is

\[
\mathcal{L}_{sup}=0.125\mathcal{L}_{focal}+0.875\mathcal{L}_{dice}
+0.25\mathcal{L}_{ce}+0.08\mathcal{L}_{rc},
\tag{22}
\]

where focal loss and cross-entropy operate on logits, Dice operates on softmax probabilities, and \(\mathcal{L}_{rc}\) is the implemented image-aware local region-consistency penalty. The configured IoU-prediction term has coefficient zero.

For epochs in which the corresponding warm-up conditions are satisfied, the code optimizes

\[
\begin{aligned}
\mathcal{L}={}&\mathcal{L}_{sup}+\lambda_{mt}(e)\mathcal{L}_{mt}
+0.05\mathcal{L}_{reg}+0.02\mathcal{L}_{intra}+0.01\mathcal{L}_{sep}\\
&+0.05\mathcal{L}_{mem}^{logit}+0.05\mathcal{L}_{mem}^{feat}
+0.20\mathcal{L}_{b}+0.10\mathcal{L}_{d}+0.10\mathcal{L}_{hard}
+0.05\widetilde{\mathcal{L}}_{geo-u}.
\end{aligned}
\tag{23}
\]

Here \(\lambda_{mt}(e)\) follows the implemented exponential warm-up over the first 10 epochs. The prototype terms start at epoch 5; memory and geometry terms start at epoch 10. The final term \(\widetilde{\mathcal{L}}_{geo-u}\) reflects the current code path but is intentionally marked with a tilde because its student and teacher boundary maps do not come from matched image views; it should not be treated as a valid semi-supervised geometry objective until corrected.

Gradients from all differentiable loss terms update the student UNI neck, fusion neck, normalized gating projections, adapter, SAM prompt encoder, mask decoder, geometry heads, and high-resolution refiner. The SAM2 encoder and UNI2-h Transformer remain frozen. Teacher inference, prototype updates, and FIFO-memory updates are performed without gradients. The optimizer is AdamW over all parameters with `requires_grad=True`; prototype and memory states are buffers rather than optimizer parameters.

## Symbol consistency table

| Symbol | Definition | Code counterpart |
|---|---|---|
| \(F^S,F^U\) | aligned SAM2/UNI-neck features | `image_embeddings`, `ex_embed` before concat |
| \(F^f,F\) | concat-neck feature / residual fused feature | `fusion_neck`, `spherical_fusion` output |
| \(G\) | spatial gate | `fusion_gate`, shape `[B,1,64,64]` |
| \(L_{low},L_{ref}\) | decoder logits before/after refiner | `low_res_masks` before/after `high_res_refiner` |
| \(p_t,q\) | teacher softmax and max probability | `teacher_probs`, `teacher_conf` |
| \(P_c\) | class prototype | `foundation_prototypes[c]` |
| \(M_{c,k}\) | FIFO instance-memory token | `instance_memory[c,k]` |
| \(\tau_p,\tau_m\) | prototype/memory temperature | `proto_temperature`, `memory_temperature` |
| \(\Omega_{mt},\Omega_p,\Omega_h\) | consistency, pseudo-region, hard-boundary valid sets | `valid`, `teacher_valid`, `hard` |
| \(B,D\) | boundary band and three foreground SDFs | `geometry_targets()` |

## Remaining code-verification and correction items

1. Correct `labels==3` to `labels==2` if the intention is vessel-specific boundary weighting; otherwise rename the setting as tubule weighting.
2. Either wire memory-specific thresholds into `update_instance_memory()` and `structure_aware_memory_loss()` or remove them from the formal configuration and manuscript.
3. Remove `broadcast_coef` or implement the corresponding loss before describing it.
4. Correct the unlabeled geometry term to compare teacher weak-view and student strong-view predictions of the same unlabeled image, with any required geometric alignment; otherwise set its coefficient to zero and omit it from the final objective.
5. Validate at runtime the selected highest-resolution FPN shape and the exact low-resolution mask-logit shape for the installed SAM2 checkpoint.
6. Do not claim erosion-based reliable cores, boundary ownership, topology preservation, cross-attention, or intrinsic Riemannian fusion unless they are separately implemented and verified.
