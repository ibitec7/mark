import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

open Real

/--
Theorem: MaRK Parameter Modulations are Strictly Bounded.
For any base parameter θ_base, learnable scale α > 0, and arbitrary unbounded MLP output x,
the adapted parameter is strictly confined within the hyper-rectangle [θ_base - α, θ_base + α].
This formally guarantees the existence of finite LMI testing vertices for Affine Quadratic Stability
-/
theorem mark_parameter_strictly_bounded (theta_base alpha x : ℝ) (h_alpha : alpha > 0) :
  theta_base - alpha < theta_base + alpha * tanh x ∧
  theta_base + alpha * tanh x < theta_base + alpha := by
  -- Mathlib theorems inherently defining the range of tanh over Reals
  have h_tanh_lower : -1 < tanh x := Real.neg_one_lt_tanh x
  have h_tanh_upper : tanh x < 1 := Real.tanh_lt_one x
  constructor
  · -- Proof 1: Lower bound
    have h1 : -alpha < alpha * tanh x := calc
      -alpha = alpha * (-1) := by ring
      _ < alpha * tanh x := mul_lt_mul_of_pos_left h_tanh_lower h_alpha
    linarith
  · -- Proof 2: Upper bound
    have h2 : alpha * tanh x < alpha := calc
      alpha * tanh x < alpha * 1 := mul_lt_mul_of_pos_left h_tanh_upper h_alpha
      _ = alpha := by ring
    linarith
