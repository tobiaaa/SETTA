"""End-to-end CMA-ES fit of the tf_mpol regress line coefficients.

Master for the cached option-3 setup: instead of regressing each warp parameter
against its per-sample oracle value (marginal fit), directly maximise the corpus
mean PESQ *under the deployed map*, jointly over the 6 line coefficients
(a, s, d each a slope+icpt in act_frac; a in log space, matching _log_line).

Each candidate is one full run_da.py in adaptation.mode=cache_eval, which reloads
the frozen extraction bundle (tf_mpol_exp.CACHE_PATH) and applies the candidate
coeffs with no model forward. da.py supplies the clean refs and scores ITU PESQ.
Workers are one SLURM array per generation (worker.sh); this master submits the
array with --wait, reads each candidate's metrics.csv, and drives CMA.

The objective is deterministic (fixed corpus + SE_FIX_SHUFFLE), so a small,
near-default popsize converges fastest; IPOP restarts (popsize doubling on
convergence) spend the remaining budget probing other basins -- the "multiple
regimes" the marginal fit could not capture. Best-so-far is checkpointed every
generation, so the run is safe to kill at any point.

Usage (see cma_master.sh):
    python cma_fit.py --cma-dir /no_backups/$USER/cma_DT_AM --ckpt <local ckpt>
"""
import argparse
import glob
import logging
import math
import os
import pickle
import shutil
import subprocess
import time

import cma
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('cma_fit')

# Coefficient order fixed here and mirrored by worker.sh's positional read.
COEF_NAMES = ['a.slope', 'a.icpt', 's.slope', 's.icpt', 'd.slope', 'd.icpt']
SIGMA0 = 1.0
# Minimisation penalty for a candidate whose worker produced no readable score
# (crash / timeout). Far worse than any real -score for every objective below
# (PESQ ~[1, 4.5], STOI [0, 1], SSNR/SI-SDR in dB, rarely past 30).
PENALTY = 100.0

# Objectives the fit can maximise. `hydra` is what goes in run_da.py's
# metrics=[...] list (the Metric subclass name); `column` is what that metric
# writes into metrics.csv -- equal to the class name except where one metric
# emits several values (eval/metrics/_base.py names() vs composite.py/ratios.py).
# `floor` is the score given to a sample the metric could not evaluate, which a
# degenerate candidate can cause; it must be at or below the metric's worst real
# value so silencing part of the corpus is never rewarded.
OBJECTIVES = {
    'pesq':  dict(hydra='PESQ',      column='PESQ',   floor=1.0),
    'stoi':  dict(hydra='STOI',      column='STOI',   floor=0.0),
    'estoi': dict(hydra='ESTOI',     column='ESTOI',  floor=0.0),
    'ssnr':  dict(hydra='SSNR',      column='SSNR',   floor=-30.0),
    'sisdr': dict(hydra='SISDR',     column='SISDR',  floor=-30.0),
    # MOS-LQO, ~[1, 5] in speech mode. floor 1.0 is the metric's own answer, not
    # a guess: ViSQOL scores an all-zero estimate at exactly 1.0 rather than
    # returning NaN, so a candidate that silences the corpus is already ranked
    # last by the real metric and read_score's flooring never has to fire.
    'visqol': dict(hydra='ViSQOL',    column='ViSQOL', floor=1.0),
    'csig':  dict(hydra='Composite', column='C_sig',  floor=1.0),
    'cbak':  dict(hydra='Composite', column='C_bak',  floor=1.0),
    'covl':  dict(hydra='Composite', column='C_ovl',  floor=1.0),
}

# Per-model search space, selected with --preset so AM and CMGAN can run at the
# same time without editing this file (editing shared state mid-run is what
# killed the first fit at generation 18).
#
# The two models want opposite regimes, and the sweeps say so plainly. Fitting
# act_frac against the free-a oracle on DT: AM gets R^2 0.432 for s and 0.115
# for log a, CMGAN gets 0.027 for s and 0.406 for log a. Their fixed-a oracle
# ceilings peak in different places too (AM at a=5, CMGAN at a=10). So AM is
# fit as "constant a, s carries act_frac" and CMGAN as "s roughly flat, a
# carries act_frac"; x0 for each is the corresponding marginal refit.
#
# `cn` must name a top-level config that sets adaptation.cache_path, because
# worker.sh overrides the group to `adaptation=tf_mpol_exp`, whose own
# cache_path is null -- and the null fallback is tf_mpol_exp.CACHE_PATH, which
# points at the CMGAN bundle. So `cn='tf_mpol'` silently ran the AM presets
# against masks_DT_CMGAN.pt; the AM presets name 'tf_mpol_exp' (masks_DT_AM.pt)
# and the CMGAN ones 'tf_mpol_cmgan' (masks_DT_CMGAN.pt) for that reason.
#
# `extra` pins everything the search does NOT touch, force-added with ++ so a
# concurrent edit to the shared config cannot change what a worker deploys.
# x0 is stated in DEPLOYED (slope, icpt) coeffs -- the form that goes in the
# config -- and converted into the search space at startup. stds/bounds are in
# ANCHOR space (see below): per parameter, the line's value at act_frac 0.22 and
# at 0.51, in that parameter's own units (log for a, log for s when s.log).
PRESETS = {
    # s in log space: R^2 0.752 vs 0.712 linear at a=5, mainly by absorbing a
    # convex bend (linear residual means swing +0.064/-0.076/+0.046 across
    # act_frac quintiles, log stays inside +-0.034). a starts as a *constant*
    # log 5 (slope 0) -- the fixed-a ceiling peaks at 5 (2.3725 vs 2.3457 at
    # a=10) and per-sample a adds only +0.018 on top of that.
    'am': dict(
        x0=[0.0, 1.6094, -4.0539, 1.2970, -0.0582, 0.0211],
        # d's std is small on purpose: its deployed values span ~0.03 across the
        # whole corpus, so an anchor step of 0.08 (what the first run used) moves
        # it several times its entire useful range and every candidate lands in
        # nonsense. a and s are in log space, where 0.5 is a factor of ~1.65.
        #      log a          log s            d
        stds=[0.5, 0.5,    0.5, 0.5,     0.02, 0.02],
        low=[-1.0, -1.0,  -3.0, -3.0,    -0.3, -0.3],
        high=[4.0, 4.0,    1.5, 1.5,      0.5, 0.5],
        cn='tf_mpol_exp',
        metric='pesq',
        extra=['adaptation.regress.s.log=true',
               'adaptation.regress.s.min=0.02',
               'adaptation.regress.s.max=1.5',
               # 30 is ample here: AM's converged fit deploys a in [4.1, 5.3].
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=30.0'],
    ),
    # 'am' with every parameter on a plain linear line -- no log space anywhere.
    # Same model, corpus, objective and cache bundle; only the coordinate system
    # of a and s changes, so the converged score is directly comparable to
    # cma_DT_AM_v2's 2.3106 and answers whether the log parameterisation buys
    # anything end-to-end or was only a better *marginal* fit.
    #
    # x0 is that converged fit re-expressed linearly: each of a and s is the
    # straight line through its own value at the two anchors, so x0 sits in the
    # same basin and the two runs start from the same deployed map at act_frac
    # 0.22/0.51. Away from the anchors they differ, which is exactly the bend
    # the log fit was absorbing -- at act_frac 0.68 linear s reads 0.113 against
    # log's 0.351, so CMA has something real to correct.
    #
    # stds/bounds are the log preset's translated into raw units around that
    # start: a's log std of 0.5 is a factor ~1.65, i.e. ~2.5 at a ~ 5, and its
    # bounds are just the clamp itself; s's 0.3 matches the linear-s presets
    # ('am_stoi' 0.25, 'cmgan_stoi' 0.3) rather than the log one. d is untouched
    # -- it was already linear in 'am' and its useful range is ~0.03 wide.
    'am_lin': dict(
        x0=[-2.4850, 5.7310, -2.7693, 1.9957, 0.0656, -0.0128],
        #      a (linear)      s (linear)      d
        stds=[2.0, 2.0,    0.3, 0.3,      0.02, 0.02],
        low=[0.1, 0.1,    -0.5, -0.5,     -0.3, -0.3],
        high=[30.0, 30.0,  2.0, 2.0,       0.5, 0.5],
        cn='tf_mpol_exp',
        metric='pesq',
        extra=['adaptation.regress.a.log=false',
               'adaptation.regress.s.log=false',
               # s.min 0.0, not 'am''s 0.02: that floor only existed because a
               # log clamp needs min > 0.
               'adaptation.regress.s.min=0.0',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=30.0'],
    ),
    # ABLATION: no act_frac dependence at all -- every parameter a single
    # constant, fitted end-to-end exactly like 'am'. This is the baseline the
    # affine map has to beat: it keeps the per-sample rank matching (the warp
    # still acts on each file's own quantile statistic) and removes only the
    # blind feature's influence on the warp SHAPE, so the gap to 'am' is what
    # act_frac itself is worth end-to-end.
    #
    # Constant presets state ONE entry per parameter, not two: a constant is
    # v_lo == v_hi in anchor space, so the search is 3-D and to_coefs emits
    # slope exactly 0. x0 stays in 6 deployed coeffs like every other preset and
    # is projected onto its value at the anchor midpoint (act_frac 0.365).
    #
    # x0 = the converged 'am' fit (cma_DT_AM_v2, PESQ 2.3106), so the ablation
    # starts from the map it ablates. Projected it deploys a 4.811, s 0.899,
    # d 0.0111. On AM only s really moves across the corpus (a spans 4.09-5.34,
    # d -0.002-0.032, s 0.352-1.500), so this is in effect an s ablation --
    # the mirror of 'cmgan_const', where a is the parameter that varies.
    #
    # stds/bounds/clamps are 'am''s, one per parameter rather than per anchor,
    # so the constant is searched in the same units and the same box.
    'am_const': dict(
        constant=True,
        x0=[-0.5161, 1.7592, -2.9853, 0.9835, 0.0656, -0.0128],
        #     log a  log s     d
        stds=[0.5,   0.5,      0.02],
        low=[-1.0,  -3.0,     -0.3],
        high=[4.0,   1.5,      0.5],
        cn='tf_mpol_exp',
        metric='pesq',
        extra=['adaptation.regress.s.log=true',
               'adaptation.regress.s.min=0.02',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=30.0'],
    ),
    # Same model as 'am', different objective -- and the objective moves both the
    # optimum and the right coordinate system, so it needs its own preset rather
    # than just a --metric flag. On sweep_blind_stoi_DT.csv the knee sits far
    # lower than PESQ's (s median 0.491 vs 1.080), and down there a LINEAR s fits
    # better than a log one (R^2 0.362 vs 0.169) -- the exact reverse of PESQ,
    # where log won because s sat near 1. So s.log is off here.
    #
    # x0 is the marginal fit from that sweep, which pinned a=10 (it is an
    # oracle_fixed_a sweep). There is no free-a STOI sweep, so a starts as the
    # constant log 10 with zero slope and CMA decides whether it should vary --
    # the same honest starting point 'am' used for PESQ.
    'am_stoi': dict(
        x0=[0.0, 2.3026, -1.9614, 1.1967, 0.2377, 0.0806],
        #      log a           s (linear)      d
        stds=[0.5, 0.5,    0.25, 0.25,    0.05, 0.05],
        low=[-1.0, -1.0,   -0.5, -0.5,    -0.3, -0.3],
        high=[4.0, 4.0,     2.0, 2.0,      0.5, 0.5],
        cn='tf_mpol_exp',
        metric='stoi',
        extra=['adaptation.regress.s.log=false',
               'adaptation.regress.s.min=0.0',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=30.0'],
    ),
    # SI-SDR on AM. Scale-invariant by construction (eval/metrics/sisdr.py
    # rescales the target onto the estimate), so the loudness renorm is invisible
    # to it and the search sees warp SHAPE only -- the reason it is a cleaner
    # objective here than SSNR, which is not scale-invariant and would spend the
    # fit partly chasing a level the renorm pins anyway.
    #
    # d's clamp is widened to [-1, 0.5], far past the [-0.1, 0.5] the other
    # presets use, because the earlier SI-SDR oracle drove d onto its aggressive
    # bound -- and a parameter pinned at a bound is exactly what produced AM's
    # degenerate PESQ fit. Here that widening is expressive, not just defensive:
    # d < 0 makes (1-d)*sigmoid(a*(q-s)) + d cross zero at q = s - ln(-1/d... )/a
    # and clamp(0,1) hard-zeroes everything below, i.e. a gate. SI-SDR rewards
    # gating noise-only bins, so it should be allowed to ask for one. d's std is
    # 0.15 rather than AM/PESQ's 0.02 for the same reason: it has real distance
    # to travel. a.max is 100, not 30, since SI-SDR's preferred steepness is
    # unknown and censoring an unknown regime is the mistake to avoid.
    #
    # x0 = AM's converged PESQ fit, since no SI-SDR sweep exists to seed from.
    # That makes the run directly readable as "how far does the optimum move
    # between objectives", which is the interesting question anyway.
    'am_sisdr': dict(
        x0=[-0.5161, 1.7592, -2.9853, 0.9835, 0.0656, -0.0128],
        #      log a          log s            d
        stds=[0.5, 0.5,    0.5, 0.5,     0.15, 0.15],
        low=[-1.0, -1.0,  -3.0, -3.0,    -1.2, -1.2],
        high=[4.7, 4.7,    1.5, 1.5,      0.6, 0.6],
        cn='tf_mpol_exp',
        metric='sisdr',
        extra=['adaptation.regress.s.log=true',
               'adaptation.regress.s.min=0.02',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0',
               'adaptation.regress.d.min=-1.0',
               'adaptation.regress.d.max=0.5'],
    ),
    # ViSQOL (MOS-LQO) on AM. No visqol sweep exists, so like 'am_sisdr' and
    # 'am_stoi' this seeds x0 from AM's converged PESQ fit (cma_DT_AM_v2, 2.3106)
    # and is read as "how far does the optimum move between objectives".
    #
    # s stays in LOG space, matching 'am' rather than 'am_stoi'. am_stoi went
    # linear on evidence -- its own sweep put the knee at s median 0.491, where a
    # linear line fits better (R^2 0.362 vs 0.169). There is no equivalent
    # evidence here, and inventing a coordinate change on a guess would make this
    # run non-comparable to the PESQ one for no reason. Log s costs ~0.004 PESQ
    # end-to-end in the other direction (am_lin vs am), so it is not a free swap.
    #
    # d is widened up front instead of after the fact: BOTH non-PESQ AM fits
    # wanted d far outside PESQ's range (am_stoi -0.542 slope, am_sisdr -0.485),
    # and am_stoi is truncated rather than converged because 21.1% of DT sat on
    # its -0.1 floor. So d.min goes to -1.0 and d's std to 0.10 (PESQ's 0.02 is
    # sized to d's ~0.03 deployed span, which only holds for PESQ). Unlike
    # 'am_sisdr' this is not an invitation to gate -- ViSQOL scores perceptual
    # similarity to the clean reference, so zeroing bins should read as
    # distortion, not as noise removal. The widened floor is there to let d move;
    # if the gate regime is bad, candidates that drift there are punished and CMA
    # keeps d >= 0 on its own, which is self-correcting where a clamp is not.
    #
    # a.max is 100, not 'am''s 30. That 30 is justified in 'am' by where PESQ's
    # converged fit actually lands (a in [4.1, 5.3]); ViSQOL's preferred
    # steepness is unknown, and censoring an unknown regime is the mistake to
    # avoid -- the same argument 'am_sisdr' and 'am_stoi' make.
    'am_visqol': dict(
        x0=[-0.5161, 1.7592, -2.9853, 0.9835, 0.0656, -0.0128],
        #      log a          log s            d
        stds=[0.5, 0.5,    0.5, 0.5,     0.10, 0.10],
        low=[-1.0, -1.0,  -3.0, -3.0,    -1.2, -1.2],
        high=[4.7, 4.7,    1.5, 1.5,      0.6, 0.6],
        cn='tf_mpol_exp',
        metric='visqol',
        extra=['adaptation.regress.s.log=true',
               'adaptation.regress.s.min=0.02',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0',
               'adaptation.regress.d.min=-1.0',
               'adaptation.regress.d.max=0.5'],
    ),
    # Mirror image of AM: a carries act_frac (R^2 0.406) and s is flat (0.027),
    # so a is the coordinate that must not be censored -- 4.2% of the free-a
    # oracle sits at the search bound of 30, which is CMGAN's version of the trap
    # that s.max=1.0 set for AM. a.max is raised to 100 for that reason. s stays
    # linear (log buys nothing: R^2 0.027 -> 0.033) and its 1.5 ceiling is never
    # reached by the oracle, so it is safe as is.
    #
    # x0 = the marginal fits from ONE sweep (sweep_blind_pesq_DT_oracle.csv, free
    # a), rather than mixing an a-line from the ablation with s/d from the a=10
    # sweep. That matters for d: the a=10 sweep gives it a +0.216 slope and the
    # free-a sweep -0.176, and with R^2 ~0.05 either way the sign is noise -- so
    # take both from the context the fit will actually run in.
    'cmgan': dict(
        x0=[-4.1344, 3.6750, -0.1094, 1.0088, -0.1757, 0.1532],
        #      log a           s (linear)      d
        stds=[0.5, 0.5,    0.25, 0.25,    0.05, 0.05],
        low=[-1.0, -1.0,   -0.5, -0.5,    -0.3, -0.3],
        high=[4.7, 4.7,     2.0, 2.0,      0.5, 0.5],   # 4.7 in log space ~ a=110
        cn='tf_mpol_cmgan',
        metric='pesq',
        extra=['adaptation.regress.s.log=false',
               'adaptation.regress.s.min=0.0',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0'],
    ),
    # 'cmgan' with a linear too -- the CMGAN counterpart of 'am_lin'. Only ONE
    # parameter changes here, not two: CMGAN's s and d were already linear (log
    # bought s nothing on this model, R^2 0.027 -> 0.033), so a is the whole
    # experiment. It is also the parameter that carries act_frac on CMGAN
    # (R^2 0.406 vs s's 0.027), which makes this the sharper test of the two.
    #
    # x0 is NOT the anchor-through linearisation 'am_lin' uses. CMGAN's a-line
    # is far steeper in log space (slope -4.10 vs AM's -0.52), so exp() of it is
    # strongly convex and a straight line through the two anchors overshoots
    # downward past the data: it reads -1.95 at act_frac 0.68, i.e. floored by
    # a.min before the search even starts. Least squares over the empirical DT
    # act_frac distribution instead:
    #
    #     af       p5     p25    p50    p75    p95
    #     a log    18.77  15.11  11.78   9.02   5.82
    #     a LS     17.46  15.13  12.46   9.60   4.90
    #
    # which tracks the log curve across the bulk and clamps for only 0.3% of the
    # corpus (act_frac p95 is 0.506, so the divergent tail is thinly populated).
    # s and d are the converged 'cmgan' fit unchanged -- they are already in the
    # coordinate system this run deploys.
    #
    # a's two anchor stds are DIFFERENT, unlike every other preset. In log space
    # one std covered both ends because a std is a ratio there; in raw units the
    # anchors sit at ~17.5 and ~4.7, and a step big enough to matter at the low
    # anchor is most of the value of the high one. 6.0/2.5 is roughly the log
    # preset's 0.5 (a factor ~1.65) evaluated at each anchor.
    'cmgan_lin': dict(
        x0=[-44.0265, 27.1619, 0.0689, 0.9673, -0.2363, 0.1637],
        #     a (linear)       s (linear)      d
        stds=[6.0, 2.5,    0.25, 0.25,    0.05, 0.05],
        low=[0.1, 0.1,    -0.5, -0.5,    -0.3, -0.3],
        high=[100.0, 100.0, 2.0, 2.0,     0.5, 0.5],
        cn='tf_mpol_cmgan',
        metric='pesq',
        extra=['adaptation.regress.a.log=false',
               'adaptation.regress.s.log=false',
               'adaptation.regress.s.min=0.0',
               'adaptation.regress.s.max=1.5',
               # a.min stays 0.1 rather than being raised to keep the LS line
               # off its floor: 0.3% of the corpus is not worth censoring the
               # coordinate that holds this model's signal.
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0'],
    ),
    # ABLATION: the CMGAN counterpart of 'am_const' -- every parameter a single
    # constant, no act_frac dependence. See 'am_const' for the 3-D constant
    # convention (one entry per parameter, x0 projected to act_frac 0.365).
    #
    # x0 = the converged 'cmgan' fit (cma_DT_CMGAN, PESQ 3.0377); projected it
    # deploys a 10.37, s 0.992, d 0.0774.
    #
    # This ablates a DIFFERENT parameter than 'am_const' does, which is the
    # point of running both. Across DT the 'cmgan' map deploys a over 2.86-23.93
    # (8.4x) while s sits at 0.978-1.014 -- dead flat -- so here the constant
    # removes a's variation, where on AM it removes s's. Expect CMGAN to lose
    # more: a is the coordinate carrying this model's act_frac signal
    # (R^2 0.406 vs s's 0.027).
    'cmgan_const': dict(
        constant=True,
        x0=[-4.1049, 3.8371, 0.0689, 0.9673, -0.2363, 0.1637],
        #     log a  s (lin)  d
        stds=[0.5,   0.25,    0.05],
        low=[-1.0,  -0.5,    -0.3],
        high=[4.7,   2.0,     0.5],
        cn='tf_mpol_cmgan',
        metric='pesq',
        extra=['adaptation.regress.s.log=false',
               'adaptation.regress.s.min=0.0',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0'],
    ),
    # SI-SDR on CMGAN. Same objective as 'am_sisdr', but the search is much
    # worse conditioned here and needs a second seed to get off the ground.
    #
    # d < 0 is what turns the map into a gate: it zeroes every rank below
    # q0 = s + ln(-d)/a. On AM that is harmless to explore, because AM's PESQ
    # optimum has a ~ 5 and s well under 1, so a small negative d gates a small
    # slice. CMGAN's PESQ optimum is the opposite corner -- s ~ 0.99 with a from
    # 5.8 to 18.8 -- and there the SAME d is catastrophic:
    #
    #     d=-0.1 gates 61-86% of bins, d=-0.55 gates 90-95%
    #
    # So every candidate that nudges d negative from x0 scores terribly, and CMA
    # learns "d must stay positive" many generations before it could discover
    # that d < 0 is only good in conjunction with a much lower s. The two regimes
    # are separated by a barrier, not a slope, and the anchor reparameterisation
    # does not help: it decorrelates slope from icpt WITHIN a parameter, not
    # s from d across them.
    #
    # Hence `seeds`: a second injected reference point in the gate regime, so
    # generation 0 measures both corners and the mean moves toward whichever
    # actually wins. It keeps CMGAN's own a-line (its steepness map is the
    # best-established thing about this model) and drops s to a flat 0.30 with
    # d = -0.30, which gates 24%/20%/9% across act_frac p5/p50/p95 -- mild, and
    # already tapering the right way purely from a's negative slope.
    #
    # s's std is 0.4 rather than 'cmgan''s 0.25 for the same reason: it has to be
    # able to travel from ~1.0 to ~0.3, which is 1.75 sigma at 0.4 and 2.8 at
    # 0.25. s stays LINEAR, matching 'cmgan', so the two runs stay comparable.
    'cmgan_sisdr': dict(
        x0=[-4.1049, 3.8371, 0.0689, 0.9673, -0.2363, 0.1637],
        seeds=[[-4.1049, 3.8371, 0.0, 0.30, 0.0, -0.30]],
        #      log a           s (linear)      d
        stds=[0.5, 0.5,    0.4, 0.4,      0.15, 0.15],
        low=[-1.0, -1.0,  -0.5, -0.5,     -1.2, -1.2],
        high=[4.7, 4.7,    2.0, 2.0,       0.6, 0.6],
        cn='tf_mpol_cmgan',
        metric='sisdr',
        extra=['adaptation.regress.s.log=false',
               'adaptation.regress.s.min=0.0',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0',
               'adaptation.regress.d.min=-1.0',
               'adaptation.regress.d.max=0.5'],
    ),
    # STOI on CMGAN. x0 is CMGAN's converged PESQ fit, as for 'cmgan_sisdr'.
    #
    # Two clamps are widened relative to 'cmgan', both to avoid repeating the
    # censoring that 'am_stoi' hit: its d sat on the -0.1 floor for 21.1% of DT
    # and its s on the 0.0 floor for 7.9%, so that fit is truncated rather than
    # converged and its numbers are provisional. d.min goes to -1.0 and s.min to
    # -0.5 here. s < 0 is meaningful, not just slack: it puts the knee left of
    # the data so f(0) is high, i.e. near-passthrough.
    #
    # NO gate seed, unlike 'cmgan_sisdr'. AM's STOI optimum has q0 = 0.000 across
    # the whole corpus -- STOI wanted d < 0 only to lower the floor, never to
    # gate -- so the gate basin is very unlikely to be where STOI lives and a
    # seed there would waste a generation-0 candidate. The widened d.min is
    # there to let d move, not to invite gating. On CMGAN the two are harder to
    # separate than on AM (at s ~ 0.99 any negative d gates immediately, see
    # 'cmgan_sisdr'), but that barrier now works in our favour: if STOI does not
    # want a gate, candidates that drift d negative are punished and CMA keeps
    # d >= 0 on its own. That is self-correcting, and unlike a -0.1 floor it is
    # not censoring.
    #
    # d's std is 0.10 rather than 'cmgan''s 0.05: am_stoi's d wanted to travel
    # ~0.2 from its start, which is 4 sigma at 0.05 and 2 at 0.10.
    'cmgan_stoi': dict(
        x0=[-4.1049, 3.8371, 0.0689, 0.9673, -0.2363, 0.1637],
        #      log a           s (linear)      d
        stds=[0.5, 0.5,    0.3, 0.3,      0.10, 0.10],
        low=[-1.0, -1.0,  -0.8, -0.8,     -1.2, -1.2],
        high=[4.7, 4.7,    2.0, 2.0,       0.6, 0.6],
        cn='tf_mpol_cmgan',
        metric='stoi',
        extra=['adaptation.regress.s.log=false',
               'adaptation.regress.s.min=-0.5',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0',
               'adaptation.regress.d.min=-1.0',
               'adaptation.regress.d.max=0.5'],
    ),
    # ViSQOL on CMGAN. x0 is CMGAN's converged PESQ fit, as for 'cmgan_sisdr'
    # and 'cmgan_stoi'; the space is 'cmgan_stoi''s, for the same reasons.
    #
    # s stays LINEAR, matching 'cmgan' -- CMGAN's s is the flat parameter
    # (R^2 0.027, and log buys nothing: 0.033), so there is no bend to absorb
    # and keeping it linear keeps this run comparable to the PESQ fit.
    #
    # NO gate seed, unlike 'cmgan_sisdr'. That seed exists because SI-SDR
    # genuinely rewards zeroing noise-only bins, so the gate basin had to be
    # measured. ViSQOL does not: it scores perceptual similarity to the clean
    # reference, and at CMGAN's PESQ optimum (s ~ 0.99) even d = -0.1 gates
    # 61-86% of bins, which should read as gross distortion. Spending a
    # generation-0 candidate there is very likely waste. d.min is still opened to
    # -1.0 so d can *move* -- the barrier around the gate regime then does the
    # censoring for us, and unlike a -0.1 floor it does so on measured scores.
    'cmgan_visqol': dict(
        x0=[-4.1049, 3.8371, 0.0689, 0.9673, -0.2363, 0.1637],
        #      log a           s (linear)      d
        stds=[0.5, 0.5,    0.3, 0.3,      0.10, 0.10],
        low=[-1.0, -1.0,  -0.8, -0.8,     -1.2, -1.2],
        high=[4.7, 4.7,    2.0, 2.0,       0.6, 0.6],
        cn='tf_mpol_cmgan',
        metric='visqol',
        extra=['adaptation.regress.s.log=false',
               'adaptation.regress.s.min=-0.5',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=100.0',
               'adaptation.regress.d.min=-1.0',
               'adaptation.regress.d.max=0.5'],
    ),
    # --- VoiceBank transfer (quick ablation; AM only) ---------------------
    # Same two fits as 'am'/'am_const' but calibrated on VoiceBank instead of
    # DAPS+TAU, to ask whether the constant ablation's gap survives a change of
    # calibration corpus. Both pin cache_path in `extra` rather than naming a
    # new top-level config: worker.sh ++-applies overrides.txt, so the bundle
    # travels with the preset and no config/tf_mpol_vbd.yaml is needed.
    #
    # ANCHORS are deliberately left at DT's 0.22/0.51 even though they are VBD's
    # p2/p88 rather than its p5/p95 (VBD act_frac runs 0.191-0.649, median 0.361,
    # against DT's 0.161-0.679, median 0.334). Keeping them identical is what
    # makes the VBD and DT fits comparable coordinate-for-coordinate; the shift
    # is small enough that both anchors stay well inside the data.
    #
    # x0 is each fit's DT counterpart, so the converged distance from x0 reads
    # directly as "how far the optimum moves between corpora".
    'am_vbd': dict(
        x0=[-0.5161, 1.7592, -2.9853, 0.9835, 0.0656, -0.0128],
        #      log a          log s            d
        stds=[0.5, 0.5,    0.5, 0.5,     0.02, 0.02],
        low=[-1.0, -1.0,  -3.0, -3.0,    -0.3, -0.3],
        high=[4.0, 4.0,    1.5, 1.5,      0.5, 0.5],
        cn='tf_mpol_exp',
        metric='pesq',
        testset='voicebank',
        extra=['adaptation.cache_path=/no_backups/m153/embs/masks_VBD_AM.pt',
               'adaptation.regress.s.log=true',
               'adaptation.regress.s.min=0.02',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=30.0'],
    ),
    # The constant ablation on VoiceBank. Clamps are 'am_const''s unchanged --
    # including s.max 1.5, which the DT constant fit sits on. Raising it here
    # would fix that censoring but make the two constants incomparable, and the
    # comparison is the point of the run.
    'am_vbd_const': dict(
        constant=True,
        x0=[0.0, 1.1593, 0.0, 0.4192, 0.0, -0.0034],
        #     log a  log s     d
        stds=[0.5,   0.5,      0.02],
        low=[-1.0,  -3.0,     -0.3],
        high=[4.0,   1.5,      0.5],
        cn='tf_mpol_exp',
        metric='pesq',
        testset='voicebank',
        extra=['adaptation.cache_path=/no_backups/m153/embs/masks_VBD_AM.pt',
               'adaptation.regress.s.log=true',
               'adaptation.regress.s.min=0.02',
               'adaptation.regress.s.max=1.5',
               'adaptation.regress.a.min=0.1',
               'adaptation.regress.a.max=30.0'],
    ),
}
# Pinned for both presets (d is never log). a's clamp is per-preset because it is
# CMGAN's carrying parameter and needs the headroom; a.min must stay > 0 either
# way, because _log_line clamps in log space.
COMMON_OVERRIDES = ['adaptation.regress.d.log=false']
# d's clamp is per-preset: [-0.1, 0.5] everywhere except the SI-SDR fit, which
# needs the floor opened up (see 'am_sisdr'). Appended after each preset's own
# `extra`, so a preset that states d.min/d.max wins.
D_CLAMP_DEFAULT = ['adaptation.regress.d.min=-0.1',
                   'adaptation.regress.d.max=0.5']

# --- anchor reparameterisation -------------------------------------------
# CMA searches each line by its VALUES at two act_frac anchors rather than by
# (slope, icpt). act_frac lives in [0.16, 0.68] on DT and never approaches 0, so
# the intercept is an anchor extrapolated far outside the data: moving the slope
# drags the line across the whole corpus unless icpt compensates, leaving the
# two coordinates strongly anti-correlated in the objective. CMA starts from an
# axis-aligned covariance (`stds`) and burns generations learning that rotation.
# Anchoring inside the data (DT p5/p95) decorrelates the axes up front and makes
# the stds and bounds above read directly in output units.
#
# Master-only: candidates are converted back to (slope, icpt) before cands.tsv
# is written, so worker.sh, the configs and _line/_log_line are all untouched.
AF_LO, AF_HI = 0.22, 0.51


def anchors_to_coefs(v):
    """Per-parameter (v_lo, v_hi) -> (slope, icpt). Inverse of coefs_to_anchors."""
    v = np.asarray(v, dtype=float).reshape(-1, 2)
    slope = (v[:, 1] - v[:, 0]) / (AF_HI - AF_LO)
    return np.stack([slope, v[:, 0] - slope * AF_LO], axis=1).ravel()


def coefs_to_anchors(c):
    """Per-parameter (slope, icpt) -> the line's values at (AF_LO, AF_HI)."""
    c = np.asarray(c, dtype=float).reshape(-1, 2)
    return np.stack([c[:, 0] * AF_LO + c[:, 1],
                     c[:, 0] * AF_HI + c[:, 1]], axis=1).ravel()


def anchor_box_to_coef_box(low, high):
    """Smallest (slope, icpt) box containing an anchor-space box.

    The anchors->coeffs map is linear but not axis-aligned, so a box maps to a
    parallelogram; enclose it by transforming the four corners of each
    parameter's 2-D anchor box. Only used for --no-anchor, which searches raw
    coefficients but should stay inside the same region.
    """
    low = np.asarray(low, float).reshape(-1, 2)
    high = np.asarray(high, float).reshape(-1, 2)
    lo, hi = [], []
    for (l0, l1), (h0, h1) in zip(low, high):
        corners = np.array([anchors_to_coefs([a, b])
                            for a in (l0, h0) for b in (l1, h1)])
        lo.extend(corners.min(axis=0))
        hi.extend(corners.max(axis=0))
    return lo, hi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cma-dir', required=True,
                   help='Run dir: per-gen candidate/score files + checkpoints.')
    p.add_argument('--ckpt', default='pretrained',
                   help='model.load for the worker (unused by cache_eval, but '
                        'run_da.py still builds the model -- use a local ckpt '
                        'to avoid a network fetch per worker).')
    p.add_argument('--preset', default='am', choices=sorted(PRESETS),
                   help='Per-model search space + pinned overrides (see PRESETS).')
    p.add_argument('--metric', default=None, choices=sorted(OBJECTIVES),
                   help='Objective to maximise. Defaults to the preset\'s. The '
                        'cache bundle is metric-independent, so switching needs '
                        'no re-cache -- but the preset x0/coordinates are tuned '
                        'per objective, so prefer a matching preset.')
    p.add_argument('--testset', default=None,
                   help='Calibration corpus; MUST match the cache pass. '
                        'Defaults to the preset\'s, or daps_tau.')
    p.add_argument('--cn', default=None,
                   help='Hydra top-level config. Defaults to the preset\'s.')
    p.add_argument('--no-anchor', dest='anchor', action='store_false',
                   help='Search raw (slope, icpt) instead of the decorrelated '
                        'act_frac-anchored values. Only changes conditioning; '
                        'the deployed coeffs are identical either way.')
    p.add_argument('--worker', default=os.path.join(os.path.dirname(__file__),
                                                    'worker.sh'))
    p.add_argument('--popsize', type=int, default=8,
                   help='Initial population (n=6 default is ~9). Doubles each '
                        'IPOP restart.')
    p.add_argument('--maxpar', type=int, default=8,
                   help='Max concurrent array tasks (SLURM --array %%maxpar).')
    p.add_argument('--restarts', type=int, default=20,
                   help='IPOP restarts after convergence (best-so-far kept).')
    p.add_argument('--incpopsize', type=int, default=2)
    p.add_argument('--sigma0', type=float, default=SIGMA0)
    p.add_argument('--resume', action='store_true',
                   help='Continue from cma-dir/state.pkl if present.')
    p.add_argument('--submit-retries', type=int, default=8,
                   help='Resubmit a generation this many times if the whole '
                        'array yields no readable score (scheduler outage). A '
                        'partial failure is NOT retried -- missing candidates '
                        'get the minimisation penalty and the search continues.')
    p.add_argument('--retry-wait', type=int, default=120,
                   help='Seconds between generation resubmits.')
    args = p.parse_args()
    args.space = PRESETS[args.preset]
    if args.space.get('constant'):
        # A constant preset searches 3 values, not 6 anchors, so --no-anchor has
        # nothing to switch off and mismatched array lengths would only surface
        # as a shape error deep inside CMA.
        if not args.anchor:
            p.error(f'--no-anchor is meaningless for constant preset '
                    f'{args.preset!r}: there is no slope to search.')
        if not all(len(args.space[k]) == 3 for k in ('stds', 'low', 'high')):
            p.error(f'constant preset {args.preset!r} must give stds/low/high '
                    f'one entry per parameter (a, s, d).')
    args.cn = args.cn or args.space['cn']
    args.metric = args.metric or args.space['metric']
    # Preset-owned like cn/metric: a corpus and its cache bundle come as a pair,
    # and defaulting to daps_tau would run a VBD preset against the DT bundle.
    args.testset = args.testset or args.space.get('testset', 'daps_tau')
    args.obj = OBJECTIVES[args.metric]
    return args


def read_score(csv_path, obj):
    """(mean score, n_rows) from a worker's metrics.csv; (None, 0) if unreadable.

    Unscorable samples are floored to obj['floor'] rather than dropped: a
    degenerate candidate can clamp the whole mask to 0, and most metrics return
    NaN on silence, so pandas' mean() would hand a candidate that silenced part
    of the corpus a *better* score than one that merely denoised it badly. The
    floor keeps the objective graded instead of a cliff into PENALTY.

    n_rows is returned so the caller can spot a truncated run -- a worker that
    dies partway leaves a short CSV whose mean looks perfectly healthy.
    """
    try:
        col = pd.read_csv(csv_path)[obj['column']].astype(float).to_numpy()
        bad = ~np.isfinite(col)
        if bad.any():
            logger.warning('%s: %d/%d samples unscorable -> floored to %.2f',
                           csv_path, int(bad.sum()), len(col), obj['floor'])
            col = np.where(bad, obj['floor'], col)
        val = float(col.mean())
        return (val, len(col)) if np.isfinite(val) else (None, 0)
    except Exception as e:
        logger.warning('unreadable %s (%s)', csv_path, e)
        return None, 0


def to_coefs(x, args):
    """Search-space vector -> deployed (slope, icpt) pairs in COEF_NAMES order."""
    if args.space.get('constant'):
        # A constant is just v_lo == v_hi in anchor space, so duplicating each
        # of the 3 searched values gives slope exactly 0 and icpt = the value.
        # Everything downstream still sees 6 coeffs: cands.tsv, worker.sh, the
        # configs and _line/_log_line are untouched by the ablation.
        return anchors_to_coefs(np.repeat(np.asarray(x, dtype=float), 2))
    return anchors_to_coefs(x) if args.anchor else np.asarray(x, dtype=float)


def to_search(c, args):
    """Deployed (slope, icpt) pairs -> search-space vector."""
    if args.space.get('constant'):
        # Project each line onto a constant: its value at the anchor midpoint
        # (act_frac 0.365). Taking icpt instead would extrapolate to act_frac 0,
        # far outside the data -- the very thing the anchoring exists to avoid.
        return coefs_to_anchors(c).reshape(-1, 2).mean(axis=1)
    return coefs_to_anchors(c) if args.anchor else np.asarray(c, dtype=float)


def write_overrides(args):
    """Pin every non-searched knob into <cma-dir>/overrides.txt for worker.sh.

    Rewritten each start so a resume picks up preset changes, and kept in the
    run dir so the fitted coeffs are always readable next to the space and
    clamps they were fitted under.
    """
    path = os.path.join(args.cma_dir, 'overrides.txt')
    lines = list(args.space['extra']) + list(COMMON_OVERRIDES)
    stated = {ln.split('=')[0] for ln in lines}
    lines += [ln for ln in D_CLAMP_DEFAULT if ln.split('=')[0] not in stated]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info('pinned overrides -> %s\n  %s', path, '\n  '.join(lines))
    return path


def _collect(cands, gdir, obj):
    """Per-candidate mean PESQ, with truncated runs rejected.

    A worker that dies partway through the corpus leaves a short metrics.csv
    whose mean is perfectly healthy-looking but covers only the samples it got
    to -- and since the easiest way to die early is a pathological candidate,
    that bias runs the wrong way. The full corpus length is the same for every
    candidate, so the generation's own maximum row count is the reference.
    """
    scored = [read_score(os.path.join(gdir, f'c{i}', 'denoised', 'metrics.csv'),
                         obj) for i in range(len(cands))]
    n_full = max((n for _, n in scored), default=0)
    pesqs = []
    for i, (val, n) in enumerate(scored):
        if val is not None and n < n_full:
            logger.warning('%s/c%d: %d/%d samples only -- truncated run, '
                           'discarding its score', gdir, i, n, n_full)
            val = None
        pesqs.append(val)
    return pesqs


def evaluate_generation(cands, gdir, args):
    """Submit one SLURM array (one task per candidate) and collect -PESQ.

    Returns fitnesses aligned with `cands` (index i <-> array task i <-> line i
    of cands.tsv), negated for CMA's minimisation.

    Robustness: `sbatch --wait` returns nonzero if *any* array task fails, so we
    never `check` it. A partial failure (some scores present) proceeds -- missing
    candidates get PENALTY, which just steers CMA away from that region. Only a
    *total* wipeout (0 readable scores == scheduler outage / systematic breakage)
    triggers a clean resubmit, up to --submit-retries.
    """
    os.makedirs(gdir, exist_ok=True)
    tsv = os.path.join(gdir, 'cands.tsv')
    # cands are in search space; worker.sh reads deployed (slope, icpt) pairs.
    np.savetxt(tsv, np.asarray([to_coefs(c, args) for c in cands]),
               fmt='%.10g', delimiter='\t')

    P = len(cands)
    cmd = ['sbatch', '--wait',
           '-o', os.path.join(gdir, 'slurm_%a.out'),
           f'--array=0-{P - 1}%{args.maxpar}',
           args.worker, gdir, args.ckpt, args.testset, args.cn,
           args.obj['hydra']]

    pesqs = [None] * P
    for attempt in range(args.submit_retries):
        if attempt:
            # Wipe partial per-candidate dirs so da.py's mkdir(denoised) is clean.
            for d in glob.glob(os.path.join(gdir, 'c*')):
                shutil.rmtree(d, ignore_errors=True)
            logger.warning('gen %s attempt %d/%d after total failure; wait %ds',
                           gdir, attempt + 1, args.submit_retries, args.retry_wait)
            time.sleep(args.retry_wait)
        logger.info('submit %s', ' '.join(cmd))
        subprocess.run(cmd, check=False)     # nonzero == some task failed; fine
        pesqs = _collect(cands, gdir, args.obj)
        if any(p is not None for p in pesqs):
            break
    else:
        raise RuntimeError(f'{gdir}: no candidate produced a score after '
                           f'{args.submit_retries} attempts -- investigate '
                           f'{gdir}/slurm_*.out (config/env, not the search).')

    fits = [PENALTY if p is None else -p for p in pesqs]
    ok = [p for p in pesqs if p is not None]
    logger.info('gen dir %s: %d/%d ok, best %s %.4f', gdir, len(ok), P,
                args.metric.upper(), max(ok) if ok else float('nan'))
    return fits, pesqs


def log_evals(args, restart, gen, cands, pesqs):
    # Logged in DEPLOYED coeffs, so evals.csv is directly comparable across runs
    # regardless of which search space produced it.
    path = os.path.join(args.cma_dir, 'evals.csv')
    rows = []
    for i, (c, p) in enumerate(zip(cands, pesqs)):
        row = {'restart': restart, 'gen': gen, 'cand': i,
               'metric': args.metric, 'score': p}
        row.update({n: v for n, v in zip(COEF_NAMES, to_coefs(c, args))})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def _clamp_report(args, coefs):
    """What the best coeffs deploy at the two act_frac anchors, vs their clamps.

    Every degenerate fit in this project looked healthy in the score and was only
    caught by checking the deployed values afterwards -- AM's first end-to-end run
    sat with s pinned at its ceiling for 100% of the corpus and still improved
    PESQ. Printing the anchor values next to the bounds each generation makes a
    parameter that has gone flat against a clamp visible while the run is young.
    """
    # Defaults first so a preset's own clamp wins, matching write_overrides.
    pins = dict(ln.split('=', 1) for ln in
                list(D_CLAMP_DEFAULT) + list(args.space['extra']))
    out = []
    for i, p in enumerate(('a', 's', 'd')):
        sl, ic = coefs[2 * i], coefs[2 * i + 1]
        lo, hi = (float(pins.get(f'adaptation.regress.{p}.{b}', d))
                  for b, d in (('min', '-inf'), ('max', 'inf')))
        # a is log unless a preset says otherwise; s/d are linear unless it does.
        log = pins.get(f'adaptation.regress.{p}.log',
                       'true' if p == 'a' else 'false') == 'true'
        v = [sl * AF_LO + ic, sl * AF_HI + ic]
        v = [math.exp(x) for x in v] if log else v
        at = ['*' if (x <= lo * (1 + 1e-6) or x >= hi * (1 - 1e-6)) else ''
              for x in v]
        out.append(f'{p}[{v[0]:.4g}{at[0]},{v[1]:.4g}{at[1]}]/({lo:g},{hi:g})')
    return '  '.join(out)


def save_best(args, xbest, fbest):
    """Write best.json in deployed coeffs, ready to paste into the config."""
    coefs = to_coefs(xbest, args)
    best = {'score': -float(fbest), 'metric': args.metric,
            'preset': args.preset}
    best.update({n: float(v) for n, v in zip(COEF_NAMES, coefs)})
    pd.Series(best).to_json(os.path.join(args.cma_dir, 'best.json'), indent=2)
    logger.info('best so far: %s %.4f  %s', args.metric.upper(),
                best['score'], {n: round(best[n], 4) for n in COEF_NAMES})
    # '*' marks a value sitting on its clamp at that anchor.
    logger.info('  deployed @act_frac %.2f/%.2f: %s',
                AF_LO, AF_HI, _clamp_report(args, coefs))


def make_es(x0, args, popsize):
    sp = args.space
    # Constant presets state one entry per parameter, anchor presets two; either
    # way the arrays are already in the search space's own units.
    if args.anchor or sp.get('constant'):
        stds, low, high = sp['stds'], sp['low'], sp['high']
    else:
        # Raw-coeff search: enclose the same region, and widen the stds by the
        # anchor spread since a slope step moves the line much further than an
        # anchor step of the same size.
        low, high = anchor_box_to_coef_box(sp['low'], sp['high'])
        stds = anchors_to_coefs(np.abs(sp['stds']) * np.array([-1, 1] * 3))
        stds = np.abs(stds) + 1e-3
    return cma.CMAEvolutionStrategy(x0, args.sigma0, {
        'popsize': popsize,
        'CMA_stds': list(stds),
        'bounds': [list(low), list(high)],
        'verbose': 1,
    })


def _next_gen_index(cma_dir):
    """One past the highest existing g#### dir, so a resume never reuses a dir
    (da.py's mkdir(denoised) would otherwise collide and fail every worker)."""
    idx = [int(os.path.basename(d)[1:]) for d in glob.glob(
        os.path.join(cma_dir, 'g[0-9]*')) if os.path.basename(d)[1:].isdigit()]
    return max(idx) + 1 if idx else 0


def load_state(args, pkl):
    """Return (es, gen, global_best, restart, popsize). Handles the new state
    dict and a legacy bare-es pickle (gen re-derived from the dirs, best from
    best.json)."""
    with open(pkl, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        s = obj
        # The pickled es carries a covariance in whatever space it was built in.
        # Resuming it under a different preset/parameterisation would keep
        # optimising, silently, in the wrong coordinates.
        was = (s.get('preset'), s.get('anchor'))
        now = (args.preset, args.anchor)
        if was != (None, None) and was != now:
            raise SystemExit(
                f'{pkl} was fitted as preset={was[0]} anchor={was[1]}, but this '
                f'run is preset={now[0]} anchor={now[1]}. The saved covariance '
                f'is in the old space -- use a fresh --cma-dir, or move the old '
                f'state.pkl aside to start this space from its x0.')
        logger.info('resumed state from %s (gen %d, restart %d)',
                    pkl, s['gen'], s['restart'])
        return s['es'], s['gen'], s['global_best'], s['restart'], s['popsize']
    # Legacy: bare CMAEvolutionStrategy. es.result.fbest already is the global
    # best over every told generation, so no best.json read is needed.
    es = obj
    gen = _next_gen_index(args.cma_dir)
    gbest = (np.array(es.result.xbest), es.result.fbest)
    logger.info('resumed legacy es from %s; gen->%d, best score %.4f',
                pkl, gen, -gbest[1])
    return es, gen, gbest, 0, getattr(es, 'popsize', args.popsize)


def main():
    args = parse_args()
    os.makedirs(args.cma_dir, exist_ok=True)
    write_overrides(args)
    pkl = os.path.join(args.cma_dir, 'state.pkl')
    legacy = os.path.join(args.cma_dir, 'es.pkl')

    # Everything below (es state, global_best, x0) lives in SEARCH space; only
    # cands.tsv, evals.csv and best.json are converted to deployed coeffs.
    x_start = to_search(args.space['x0'], args)
    logger.info('preset %-6s cn=%-14s space=%s\n  x0 coeffs %s\n  x0 search %s',
                args.preset, args.cn, 'anchor' if args.anchor else 'slope/icpt',
                np.round(args.space['x0'], 4).tolist(), np.round(x_start, 4).tolist())

    es = None
    global_best = (x_start.copy(), np.inf)   # (xbest, fbest); minimisation
    gen, restart, popsize, x0 = 0, 0, args.popsize, x_start.copy()
    if args.resume:
        src = pkl if os.path.exists(pkl) else (legacy if os.path.exists(legacy)
                                               else None)
        if src:
            es, gen, global_best, restart, popsize = load_state(args, src)

    while restart <= args.restarts:
        if es is None:
            es = make_es(x0, args, popsize)
            # Force x0 itself into the first population. CMA otherwise only ever
            # evaluates perturbations of it, so the reference point is never
            # measured and best.json can sit *below* the config the run started
            # from -- which is what the first s.max=1.5 run did (2.12 vs the
            # ~2.24 regressed baseline) with no way to see it from the logs.
            # Any extra reference points the preset names, in DEPLOYED coeffs
            # like x0. For a landscape whose regimes are separated by a barrier
            # rather than a slope, perturbing x0 alone never reaches the other
            # basin -- see PRESETS['cmgan_sisdr'].
            inject = [np.asarray(x0, dtype=float)]
            inject += [to_search(s, args) for s in args.space.get('seeds', ())]
            es.inject(inject, force=True)
            logger.info('restart %d: popsize %d, x0 %s (+%d seed(s), injected)',
                        restart, popsize, np.round(x0, 3).tolist(),
                        len(inject) - 1)

        while not es.stop():
            cands = es.ask()
            gdir = os.path.join(args.cma_dir, f'g{gen:04d}')
            fits, pesqs = evaluate_generation(cands, gdir, args)
            es.tell(cands, fits)
            es.disp()
            log_evals(args, restart, gen, cands, pesqs)
            if es.result.fbest < global_best[1]:
                global_best = (np.array(es.result.xbest), es.result.fbest)
            save_best(args, *global_best)
            gen += 1
            with open(pkl, 'wb') as f:
                pickle.dump({'es': es, 'gen': gen, 'global_best': global_best,
                             'restart': restart, 'popsize': popsize,
                             'preset': args.preset, 'anchor': args.anchor}, f)

        logger.info('restart %d stopped: %s', restart, es.stop())
        # IPOP: grow the population and restart from the best basin found.
        restart += 1
        popsize *= args.incpopsize
        x0 = global_best[0].copy()
        es = None

    logger.info('done. best %s %.4f', args.metric.upper(), -global_best[1])
    save_best(args, *global_best)


if __name__ == '__main__':
    main()
