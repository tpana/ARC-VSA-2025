"""
HeuristicObjObjSolver — VSA-on-ARC without the neural network.

Replaces SolutionProgram + ObjectProgramPredictor with a deterministic
property-relevance heuristic. ~10-100x faster, no approximation errors
from neural training. Foundation for HyPRA's PP inference pipeline.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from objobj_solver import *   # ObjObjSolver, N_DIMENSIONS, get_unique_parameters, etc.
import numpy as np
import copy

# ── tolerances ─────────────────────────────────────────────────────────────
_ATOL          = 1e-3   # np.allclose tolerance for parameter matching
_SIM_THRESHOLD = 1.5    # sum of (colour+centre+shape) cosines to be "same type"
                         # range is [−3, 3]; 1.5 ≈ "at least half similar"


# ── parameter-rule inference ────────────────────────────────────────────────

def _infer_param_rule(in_objs, p_vecs):
    # Force plain numpy — SSP subclass overrides operators, breaks np.allclose
    p_vecs = [np.array(p, dtype=float).flatten() for p in p_vecs]

    if not p_vecs:
        return ('none', None)

    # 1. Constant
    if all(np.allclose(p_vecs[0], p, atol=_ATOL) for p in p_vecs):
        return ('constant', p_vecs[0].copy())

    n = len(in_objs)
    if n == 0 or n != len(p_vecs):
        return ('fallback', p_vecs[0].copy())

    # Force plain numpy for property representations too
    c_vecs  = [np.array(o.get_colour_representation(), dtype=float).flatten() for o in in_objs]
    ct_vecs = [np.array(o.get_centre_representation(),  dtype=float).flatten() for o in in_objs]
    sh_vecs = [np.array(o.get_shape_representation(),   dtype=float).flatten() for o in in_objs]

    # 2. Colour identity
    if all(np.allclose(c_vecs[k],  p_vecs[k], atol=_ATOL) for k in range(n)):
        return ('colour', None)

    # 3. Centre identity
    if all(np.allclose(ct_vecs[k], p_vecs[k], atol=_ATOL) for k in range(n)):
        return ('centre', None)

    # 4. Shape identity
    if all(np.allclose(sh_vecs[k], p_vecs[k], atol=_ATOL) for k in range(n)):
        return ('shape', None)

    # 5. Fallback
    return ('fallback', p_vecs[0].copy())


def _apply_param_rule(rule_type, rule_value, obj):
    """Apply a single parameter rule to a single object."""
    if rule_type == 'constant':
        return rule_value
    elif rule_type == 'colour':
        return obj.get_colour_representation()
    elif rule_type == 'centre':
        return obj.get_centre_representation()
    elif rule_type == 'shape':
        return obj.get_shape_representation()
    else:  # 'fallback' or 'none'
        return rule_value if rule_value is not None else np.zeros(N_DIMENSIONS)


# ── HeuristicProgramPredictor ───────────────────────────────────────────────

class HeuristicProgramPredictor:
    """
    Deterministic replacement for SolutionProgram + ObjectProgramPredictor.

    Fits on training distributions from the hitting set, then applies
    deterministic rules to test objects. Output format is identical to
    SolutionProgram.forward(), so deduce_output() works unchanged.
    """

    def __init__(self, train_input, train_output):
        """
        Parameters
        ----------
        train_input  : list[list[ARCObject]]  shape [n_demos][n_objs]
        train_output : dict {op_class: [masks, params]}
            masks  : list[list[int]]           shape [n_demos][n_objs]
            params : list[list[list[ndarray]]] shape [n_demos][n_objs][n_params]
        """
        self._rules = {}
        self._fit(train_input, train_output)

    def _fit(self, train_input, train_output):
        for op_class, (masks, params) in train_output.items():
            n_params   = op_class().get_n_parameters()
            pos_objs   = []   # training objects that DID receive this op
            pos_params = []   # their parameter lists
            total      = 0

            for demo_i, demo_objs in enumerate(train_input):
                for obj_j, obj in enumerate(demo_objs):
                    total += 1
                    if masks[demo_i][obj_j]:
                        pos_objs.append(obj)
                        # Convert to plain numpy at storage time
                        converted = [
                            np.array(params[demo_i][obj_j][p_idx], dtype=float).flatten()
                            for p_idx in range(len(params[demo_i][obj_j]))
                        ]
                        pos_params.append(converted)

            # Selection rule
            n_pos = len(pos_objs)
            if n_pos == 0:
                selection = ('none',)
            elif n_pos == total:
                selection = ('all',)
            else:
                selection = ('selective', pos_objs)

            # Parameter rules — one per parameter slot
            param_rules = []
            for p_idx in range(n_params):
                p_vecs = [pos_params[k][p_idx] for k in range(n_pos)]
                param_rules.append(_infer_param_rule(pos_objs, p_vecs))

            self._rules[op_class] = {
                'selection':   selection,
                'param_rules': param_rules,
                'n_params':    n_params,
            }

    def _prob(self, obj, selection):
        """
        Return a probability/score for whether this operation applies to obj.
        For 'selective', returns the max similarity score (continuous) so that
        argmax in deduce_output() picks the best-matching object (needed for
        ExtractOperation).
        """
        kind = selection[0]
        if kind == 'all':
            return 1.0
        if kind == 'none':
            return 0.0
        # 'selective'
        pos_objs = selection[1]
        if not pos_objs:
            return 0.0
        sims = [np.sum(obj.get_similarity_to(tr)) for tr in pos_objs]
        return float(max(sims))

    def __call__(self, objects):
        """
        Predict for a list of objects (one demo/test set).

        Returns
        -------
        dict {op_class: [probs, params_per_obj]}
          probs         : list[float]              length n_objs
          params_per_obj: list[list[np.ndarray]]   shape [n_objs][n_params]
        """
        result = {}
        for op_class, rule in self._rules.items():
            probs  = [self._prob(obj, rule['selection'])            for obj in objects]
            params = [[_apply_param_rule(rt, rv, obj)
                       for rt, rv in rule['param_rules']]           for obj in objects]
            result[op_class] = [probs, params]
        return result


# ── helper: slice training predictions for one demo ─────────────────────────

def _train_preds_for_demo(train_output, demo_i):
    result = {}
    for op_class, (masks, params) in train_output.items():
        probs = [float(masks[demo_i][j]) for j in range(len(masks[demo_i]))]
        params_i = [
            [np.array(params[demo_i][obj_j][p], dtype=float).flatten()
             for p in range(len(params[demo_i][obj_j]))]
            for obj_j in range(len(params[demo_i]))
        ]
        result[op_class] = [probs, params_i]
    return result


# ── HeuristicObjObjSolver ───────────────────────────────────────────────────

class HeuristicObjObjSolver(ObjObjSolver):
    """
    Drop-in replacement for ObjObjSolver. Identical pipeline except Step 3
    uses HeuristicProgramPredictor instead of SolutionProgram.

    Training demos use known-correct answers (perfect train accuracy by
    construction). Test demos use the derived heuristic rules.
    """

    def solve_task(self):
        # ── Step 1: grid size ────────────────────────────────────────────
        self.grid_size_hypothesis, conf = self.induce_grid_size_hypothesis()
        print(f"[1] Grid size: {self.grid_size_hypothesis}  (conf={conf:.3f})")

        # ── Step 2: object hypothesis + operation abduction ──────────────
        (self.object_hypothesis,
         self.object_hypothesis_confidence,
         train_in_objs,
         train_prog_preds) = self.abduce_object_and_operation_hypotheses()
        print(f"[2] Objects:   {self.object_hypothesis(None)}  "
              f"(conf={self.object_hypothesis_confidence:.3f})")
        ops = ", ".join(op().__class__.__name__ for op in train_prog_preds)
        print(f"    Operations: {ops}")

        # ── Step 3: build heuristic predictor (no NN!) ───────────────────
        predictor = HeuristicProgramPredictor(train_in_objs, train_prog_preds)
        print(f"[3] Heuristic rules:")
        for op_class, rule in predictor._rules.items():
            sel    = rule['selection'][0]
            prules = [r[0] for r in rule['param_rules']]
            print(f"    {op_class().__class__.__name__:20s}  "
                  f"selection={sel}  params={prules}")

        # ── Step 4: generate outputs ─────────────────────────────────────
        test_in_objs, _ = self.generate_object_data(
            self.test_in_grids, self.test_out_grids, self.object_hypothesis
        )

        # Training — use known-correct predictions
        train_out_grids = []
        for i in range(len(self.train_in_grids)):
            preds = _train_preds_for_demo(train_prog_preds, i)
            train_out_grids.append(ObjObjSolver.deduce_output(
                train_in_objs[i], preds,
                self.train_in_grids[i].get_size_representation(),
                self.grid_size_hypothesis.apply(
                    self.train_in_grids[i].get_size_representation()),
            ))

        # Test — use heuristic predictions
        test_out_grids = []
        for i in range(len(self.test_in_grids)):
            preds = predictor(test_in_objs[i])
            test_out_grids.append(ObjObjSolver.deduce_output(
                test_in_objs[i], preds,
                self.test_in_grids[i].get_size_representation(),
                self.grid_size_hypothesis.apply(
                    self.test_in_grids[i].get_size_representation()),
            ))

        return self.print_results(train_out_grids, test_out_grids)