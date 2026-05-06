import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Real.StarOrdered

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/--
Definition of Affine Quadratic Stability (AQS)
A system is AQS if there exists a common Positive Definite matrix P
such that for all v, P - A(v)^T P A(v) - εI is Positive Semidefinite.
-/
def IsAQS {V : Type*} (A : V → Matrix n n ℝ) (P : Matrix n n ℝ) (ε : ℝ) : Prop :=
  P.PosDef ∧ ε > 0 ∧
  ∀ v, (P - (A v)ᵀ * P * (A v) - ε • (1 : Matrix n n ℝ)).PosSemidef

/--
Theorem: If a system satisfies Affine Quadratic Stability (AQS),
then it guarantees strict energy decay for any non-zero state x
across all parameter variations v.
-/
theorem aqs_implies_decay {V : Type*} {A : V → Matrix n n ℝ} {P : Matrix n n ℝ} {ε : ℝ}
  (h : IsAQS A P ε) :
  ∀ (v : V) (x : n → ℝ), x ≠ 0 → dotProduct x (((A v)ᵀ * P * (A v) - P) *ᵥ x) < 0 := by
  intros v x hx
  have ⟨_, hε, hpsd⟩ := h
  have h_psd := Matrix.PosSemidef.dotProduct_mulVec_nonneg (hpsd v) x
  -- Expand the Positive Semidefinite property
  rw [sub_mulVec, sub_mulVec, dotProduct_sub, dotProduct_sub] at h_psd
  -- Simplify the epsilon component
  have heps : star x ⬝ᵥ ((ε • (1 : Matrix n n ℝ)) *ᵥ x) = ε * (star x ⬝ᵥ x) := by
    rw [smul_mulVec, dotProduct_smul, one_mulVec, smul_eq_mul]
  -- Prove that the norm squared is positive
  have hx_pos : 0 < star x ⬝ᵥ x := by
    simpa only [one_mulVec] using (Matrix.PosDef.one (R := ℝ) (n := n)).dotProduct_mulVec_pos hx
  rw [heps] at h_psd
  rw [sub_mulVec, dotProduct_sub]
  have h_star : star x = x := rfl
  simp_rw [h_star] at h_psd hx_pos
  -- Conclude with linarith
  have : 0 < ε * (x ⬝ᵥ x) := mul_pos hε hx_pos
  linarith
