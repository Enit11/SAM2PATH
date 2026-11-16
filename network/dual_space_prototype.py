"""Paired Euclidean/spherical region prototypes and class-positive FIFO memory."""
import torch
from torch import nn
import torch.nn.functional as F
import cv2


class DualSpaceProjection(nn.Module):
    def __init__(self, channels=256, dim=128):
        super().__init__()
        self.euclidean = nn.Conv2d(channels, dim, 1)
        self.spherical = nn.Conv2d(channels, dim, 1)
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features):
        return self.euclidean(features).float(), F.normalize(self.spherical(features).float(), dim=1)


class PairedPrototypeMemory(nn.Module):
    def __init__(self, classes, dim, size, cfg):
        super().__init__()
        self.cfg = cfg
        if any(float(cfg.get(k, .1)) <= 0 for k in ('euclidean_temperature', 'spherical_temperature')) or size < 1:
            raise ValueError('Temperatures and memory capacity must be positive')
        self.defer_writes = False
        self.pending = []
        self.diagnostics = {}
        self.classes, self.size = classes, size
        self.register_buffer('proto_e', torch.zeros(classes + 1, dim))
        self.register_buffer('proto_s', torch.zeros(classes + 1, dim))
        self.register_buffer('ready', torch.zeros(classes + 1, dtype=torch.bool))
        self.register_buffer('memory_e', torch.zeros(classes + 1, size, dim))
        self.register_buffer('memory_s', torch.zeros(classes + 1, size, dim))
        self.register_buffer('memory_ready', torch.zeros(classes + 1, size, dtype=torch.bool))
        self.register_buffer('pointer', torch.zeros(classes + 1, dtype=torch.long))

    def scores(self, e, s, bank_e, bank_s, projection, bank_classes=None):
        # Mean squared distance avoids dimension-dependent distance scaling.
        le = -(e[None] - bank_e).square().mean(-1) / self.cfg.get('euclidean_temperature', 0.1)
        ls = (s[None] * bank_s).sum(-1) / self.cfg.get('spherical_temperature', 0.1)
        if bank_classes is not None:
            # Correct class occupancy prior separately in each geometry.
            _, inverse, counts = bank_classes.unique(return_inverse=True, return_counts=True)
            correction = counts[inverse].float().log()
            le, ls = le - correction, ls - correction
        alpha = torch.sigmoid(projection.gate(torch.cat((e, s))))
        self.diagnostics['gate_alpha_last'] = alpha.detach().mean()
        # Log-space mixture prevents underflow in low-probability classes.
        a = alpha.clamp(1e-6, 1 - 1e-6)
        return torch.logaddexp(a.log() + le.log_softmax(-1), (1-a).log() + ls.log_softmax(-1))

    def regions(self, owner, features, labels, projection, confidence=None, valid_map=None, memory=False):
        if features.requires_grad and torch.is_grad_enabled():
            features.register_hook(lambda grad: self.diagnostics.__setitem__('upstream_gradient_norm', grad.detach().float().norm()))
        ef, sf = projection(features)
        small = owner.resize_labels_for_feature(labels, features.shape[-2:])
        policy = owner.memory_cfg if memory else owner.proto_cfg
        min_pixels = int(policy.get('min_region_pixels', owner.proto_min_region_pixels))
        min_fraction = float(policy.get('min_valid_fraction', 0.9 if memory else 0.7))
        threshold = float(policy.get('conf_threshold', owner.proto_conf_threshold))
        valid = small > 0
        if valid_map is not None:
            valid &= owner.resize_labels_for_feature(valid_map.long(), features.shape[-2:]).bool()
        if confidence is not None:
            valid &= owner.resize_conf_for_feature(confidence, features.shape[-2:]) >= threshold
        for b in range(labels.shape[0]):
            # CC is extracted BEFORE confidence masking. Reliability measures
            # retained fraction of a whole candidate, not tautological class purity.
            regions = []
            label_np = small[b].detach().cpu().numpy()
            for c in range(1, self.classes + 1):
                count, components, stats, _ = cv2.connectedComponentsWithStats((label_np == c).astype('uint8'), connectivity=8)
                for k in range(1, count):
                    area = int(stats[k, cv2.CC_STAT_AREA])
                    if area >= min_pixels:
                        regions.append((area, c, components == k))
            regions.sort(key=lambda r: r[0], reverse=True)
            for area, c, mask_np in regions[:owner.proto_max_regions_per_image]:
                whole = torch.as_tensor(mask_np, device=features.device)
                mask = whole & valid[b]
                if int(mask.sum()) < min_pixels or float(mask.sum()) / area < min_fraction:
                    continue
                yield c, ef[b, :, mask].mean(-1), F.normalize(sf[b, :, mask].mean(-1), dim=0)

    @torch.no_grad()
    def update(self, owner, features, labels, confidence=None, valid_map=None, memory=False):
        projection = owner.teacher_model.dual_space_projection
        records = list(self.regions(owner, features, labels, projection, confidence, valid_map, memory))
        if self.defer_writes:
            self.pending.append((memory, confidence is not None or valid_map is not None, records))
            return features.new_tensor(0.)  # committed counts are logged after flush
        return self.write_records(owner, records, memory)

    @torch.no_grad()
    def write_records(self, owner, records, memory=False):
        projection = owner.teacher_model.dual_space_projection
        count = 0
        if not memory:
            # Equal region weighting, one EMA update per observed class per batch.
            for c in range(1, self.classes + 1):
                selected = [(e, s) for k, e, s in records if k == c]
                if not selected:
                    continue
                e = torch.stack([x[0] for x in selected]).mean(0)
                s = F.normalize(torch.stack([x[1] for x in selected]).mean(0), dim=0)
                m = owner.proto_momentum if self.ready[c] else 0.
                self.proto_e[c].lerp_(e, 1-m)
                self.proto_s[c].copy_(F.normalize(m*self.proto_s[c]+(1-m)*s, dim=0))
                self.ready[c] = True
                count += len(selected)
        else:
            self.diagnostics['memory_candidates'] = self.proto_e.new_tensor(float(len(records)))
            ids = self.ready.nonzero().flatten()
            # Require >=2 available classes for a meaningful semantic check.
            if len(ids) < 2:
                self.diagnostics['memory_acceptance'] = self.proto_e.new_tensor(0.)
                return self.proto_e.new_tensor(0.)
            for c, e, s in records:
                logq = self.scores(e, s, self.proto_e[ids], self.proto_s[ids], projection)
                if ids[logq.argmax()].item() != c or logq.max().exp() < self.cfg.get('memory_match_threshold', 0.6):
                    continue
                slot = int(self.pointer[c])
                # FIFO overwrite: slots do not identify the same historical instance.
                self.memory_e[c, slot].copy_(e)
                self.memory_s[c, slot].copy_(s)
                self.memory_ready[c, slot] = True
                self.pointer[c] = (slot + 1) % self.size
                count += 1
        if memory:
            self.diagnostics['memory_acceptance'] = self.proto_e.new_tensor(count / max(len(records), 1))
        return self.proto_e.new_tensor(float(count))

    @torch.no_grad()
    def flush(self, owner):
        # Memory validation still sees PRE-BATCH prototypes. Only afterwards
        # update prototypes once/class from weighted labelled/unlabelled means.
        memory_count = self.proto_e.new_tensor(0.)
        for memory, _, records in self.pending:
            if memory:
                memory_count += self.write_records(owner, records, True)
        merged = []
        for c in range(1, self.classes + 1):
            sources = []
            for unlabeled in (False, True):
                selected = [(e, s) for mem, u, recs in self.pending if not mem and u == unlabeled for k, e, s in recs if k == c]
                weight = float(self.cfg.get('unlabeled_update_weight', .5)) if unlabeled else 1.
                if selected and weight > 0:
                    sources.append((weight, torch.stack([r[0] for r in selected]).mean(0), torch.stack([r[1] for r in selected]).mean(0)))
            if sources:
                total = sum(w for w, _, _ in sources)
                # Never bootstrap a class exclusively from a pseudo-label.
                has_gt = any(not mem and not u and any(k == c for k, _, _ in recs) for mem, u, recs in self.pending)
                if self.ready[c] or has_gt:
                    merged.append((c, sum(w*e for w, e, _ in sources)/total, F.normalize(sum(w*s for w, _, s in sources)/total, dim=0)))
        prototype_count = self.write_records(owner, merged)
        self.pending.clear()
        self.defer_writes = False
        return prototype_count, memory_count

    @torch.no_grad()
    def reset(self):
        for buffer in self.buffers():
            buffer.zero_()
        self.pending.clear()
        self.diagnostics.clear()
        self.defer_writes = False

    def losses(self, owner, features, labels, confidence=None, valid_map=None, memory=False):
        projection = owner.model.dual_space_projection
        zero = features.sum() * 0
        match, compact, sep = [], [], []
        if memory:
            indices = self.memory_ready.nonzero()
            ids = indices[:, 0]
            be, bs = self.memory_e[self.memory_ready], self.memory_s[self.memory_ready]
        else:
            ids = self.ready.nonzero().flatten()
            be, bs = self.proto_e[ids], self.proto_s[ids]
        # Clone detached banks: later teacher writes must not invalidate autograd.
        be, bs = be.detach().clone(), bs.detach().clone()
        if ids.unique().numel() < 2:
            return zero, zero, zero, zero.detach()
        for c, e, s in self.regions(owner, features, labels, projection, confidence, valid_map, memory):
            positive = ids == c
            if not positive.any():
                continue
            logq = self.scores(e, s, be, bs, projection, ids if memory else None)
            if not memory:
                euclidean_prediction = ids[((e[None]-be).square().mean(-1)).argmin()]
                spherical_prediction = ids[(s[None]*bs).sum(-1).argmax()]
                self.diagnostics['euclidean_match_last'] = (euclidean_prediction == c).float().detach()
                self.diagnostics['spherical_match_last'] = (spherical_prediction == c).float().detach()
                self.diagnostics['dual_match_last'] = (ids[logq.argmax()] == c).float().detach()
                self.diagnostics['euclidean_token_norm_last'] = e.detach().norm()
            match.append(-torch.logsumexp(logq[positive], 0))
            # Margin acts on live student scores, not detached prototype pairs.
            if not memory:
                sep.append(F.relu(owner.proto_sep_margin + logq[~positive].max() - logq[positive].max()))
        mean = lambda xs: torch.stack(xs).mean() if xs else zero
        return mean(match), mean(compact), mean(sep), features.new_tensor(float(len(match)))
