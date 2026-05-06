import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

open Real

/-!
# Constructive ZOH Stability for MaRK-Modulated Hydra SSM

This file formalizes the constructive stability argument for the Zero-Order Hold (ZOH)
discretization chain used in the MaRK adapter framework:

  A_log' = A_log + β · tanh(ψ(c_t))     (bounded shift)
  A_continuous = -exp(A_log')             (strictly negative)
  Δ = softplus(·)                         (strictly positive)
  Ā = exp(A_continuous · Δ)               (element-wise in (0, 1))

We prove that each discrete eigenvalue ā_i satisfies 0 < ā_i < 1, guaranteeing
spectral radius ρ(Ā) < 1 and hence P = I as a valid common Lyapunov function.
-/

/--
The negation of exp(v) is strictly negative for any real v.
Formalizes: A_continuous = -exp(A_log') < 0 in the ZOH discretization chain.
-/
theorem neg_exp_is_neg (v : ℝ) : -exp v < 0 := by
  linarith [exp_pos v]

/--
exp of a negative argument is strictly between 0 and 1.
Key step: if a < 0 then 0 < exp(a) < 1.
-/
theorem exp_neg_in_unit (x : ℝ) (hx : x < 0) : 0 < exp x ∧ exp x < 1 := by
  constructor
  · exact exp_pos x
  · have h1 : exp x < exp 0 := by
      exact exp_strictMono hx
    rwa [exp_zero] at h1

/--
The ZOH discretization exp(-exp(v) · t) lies in the open unit interval (0, 1)
for any real v and any positive discretization step t > 0.

This is the core stability result: since -exp(v) < 0 and t > 0, the product
-exp(v) · t < 0, and exp of a negative number is in (0, 1).
-/
theorem zoh_in_unit_interval (v t : ℝ) (ht : 0 < t) :
    0 < exp (-exp v * t) ∧ exp (-exp v * t) < 1 := by
  have h_neg : -exp v < 0 := neg_exp_is_neg v
  have h_prod : -exp v * t < 0 := mul_neg_of_neg_of_pos h_neg ht
  exact exp_neg_in_unit _ h_prod

/--
MaRK ZOH stability theorem: For any base log-space parameter A_log, any bounded
modulation shift δ (with |δ| < β, though the bound is not needed for the stability
conclusion), and any positive discretization step t > 0, the discrete eigenvalue
exp(-exp(A_log + δ) · t) is strictly less than 1.

This guarantees ρ(Ā(c_t)) < 1 for all conditioning contexts c_t, establishing
that the identity matrix P = I is a valid common Lyapunov function.
-/
theorem mark_zoh_stable (A_log δ t : ℝ) (ht : 0 < t) :
    exp (-exp (A_log + δ) * t) < 1 :=
  (zoh_in_unit_interval (A_log + δ) t ht).2

/--
The discrete eigenvalue is also strictly positive, confirming it lies in (0, 1)
rather than just being bounded above by 1.
-/
theorem mark_zoh_positive (A_log δ t : ℝ) (ht : 0 < t) :
    0 < exp (-exp (A_log + δ) * t) :=
  (zoh_in_unit_interval (A_log + δ) t ht).1

/--
Lyapunov energy dissipation: for a scalar discrete eigenvalue ā = exp(-exp(A_log + δ) · t)
with t > 0, the Lyapunov decrease ā² - 1 is strictly negative.

This formalizes that for P = I, the Lyapunov condition (A^T P A - P)_{ii} = ā_i² - 1 < 0
holds element-wise, proving negative definiteness of the Lyapunov decrease matrix.
-/
theorem lyapunov_decrease_negative (A_log δ t : ℝ) (ht : 0 < t) :
    exp (-exp (A_log + δ) * t) ^ 2 - 1 < 0 := by
  have h_lt_one := mark_zoh_stable A_log δ t ht
  have h_pos := mark_zoh_positive A_log δ t ht
  nlinarith [sq_nonneg (exp (-exp (A_log + δ) * t))]
