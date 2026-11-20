"""Simulator, evaluation harness and the learning policies.

The most important test here is :func:`test_coupling_reproduces_the_observed_outcome`. The whole
simulator rests on the conditional coupling described in ``simulation/environment.py``: at the
exposure LendingClub actually funded, the simulation must reproduce what really happened. If it
does not, every policy comparison is being run against invented outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.decision.expected_loss import CashflowModel, EconomicsConfig, fit_cashflow_model
from credit_risk.decision.policies import (
    ActionSpace,
    ApproveAllPolicy,
    ExpectedProfitPolicy,
    FixedThresholdPolicy,
    RiskBandPolicy,
)
from credit_risk.rl.formulation import StateEncoder, analytical_shadow_price
from credit_risk.rl.policy import (
    BanditConfig,
    BudgetAwareQPolicy,
    LinUCBPolicy,
    QLearningConfig,
    ThompsonSamplingPolicy,
    train_policy,
)
from credit_risk.simulation.environment import (
    LendingEnvironment,
    SimulationConfig,
    build_applicant_pool,
)
from credit_risk.simulation.evaluation import (
    compare_policies,
    difference_table,
    evaluate_policy,
    paired_difference,
    summary_table,
)


@pytest.fixture(scope="module")
def cashflow(split_frames) -> CashflowModel:
    return fit_cashflow_model(split_frames["train"])


@pytest.fixture(scope="module")
def pool(split_frames, cashflow):
    test = split_frames["test"]
    rng = np.random.default_rng(3)
    # A crude but monotone stand-in for a model score; the simulator's behaviour is what is
    # under test, not the quality of the PD estimate.
    pd_hat = np.clip(
        0.15 + 0.12 * rng.standard_normal(len(test)) + 0.1 * (test["default"].to_numpy() - 0.15),
        0.01, 0.9,
    )
    return build_applicant_pool(test, pd_hat, cashflow)


@pytest.fixture
def space() -> ActionSpace:
    return ActionSpace().without_review()


def make_env(pool, cashflow, space, budget=None, n_steps=400) -> LendingEnvironment:
    return LendingEnvironment(
        pool, cashflow, EconomicsConfig(), space,
        SimulationConfig(n_steps=n_steps, capital_budget=budget),
    )


# ---------------------------------------------------------------------------
# The coupling
# ---------------------------------------------------------------------------

def test_coupling_reproduces_the_observed_outcome(pool, cashflow, space):
    """At the funded exposure, the simulation must reproduce reality exactly.

    Approving everyone at the amount actually funded should give a default flag identical to the
    observed one for every applicant. Any mismatch means the simulator is inventing outcomes.
    """
    env = make_env(pool, cashflow, space, n_steps=500)
    medium = space.names.index("approve_medium")
    env.reset(seed=1)
    mismatches = 0
    checked = 0
    while True:
        obs = env._observation()
        if obs["done"]:
            break
        applicant = obs["applicant"]
        observed = bool(pool.observed_default[applicant] > 0.5)
        _, done, _ = env.step(medium)
        simulated = bool(env._log[-1]["defaulted"])
        # Requested and funded amounts coincide for almost all LendingClub loans; where they do
        # not, the re-scored PD legitimately differs and the coupling may flip.
        if abs(pool.requested_amount[applicant] - pool.observed_exposure[applicant]) < 1.0:
            checked += 1
            mismatches += int(simulated != observed)
        if done:
            break
    assert checked > 100
    assert mismatches == 0, f"{mismatches}/{checked} outcomes did not reproduce reality"


def test_coupling_is_shared_across_policies(pool, cashflow, space):
    """Common random numbers: the same seed must give the same applicants and the same draws."""
    env = make_env(pool, cashflow, space)
    env.reset(seed=5)
    first_order = env._order.copy()
    first_u = env._u.copy()
    env.reset(seed=5)
    assert np.array_equal(first_order, env._order)
    assert np.allclose(first_u, env._u)


def test_different_seeds_give_different_episodes(pool, cashflow, space):
    env = make_env(pool, cashflow, space)
    env.reset(seed=1)
    first = env._order.copy()
    env.reset(seed=2)
    assert not np.array_equal(first, env._order)


# ---------------------------------------------------------------------------
# Episode mechanics
# ---------------------------------------------------------------------------

def test_rejecting_everything_earns_exactly_zero(pool, cashflow, space):
    class RejectAll(FixedThresholdPolicy):
        def __init__(self):
            super().__init__(threshold=-1.0, action_space=space)

    result = make_env(pool, cashflow, space).run(RejectAll(), seed=0)
    assert result.total_reward == pytest.approx(0.0)
    assert result.n_approved == 0
    assert result.capital_deployed == 0.0


def test_budget_is_never_exceeded(pool, cashflow, space):
    budget = 250_000.0
    env = make_env(pool, cashflow, space, budget=budget)
    result = env.run(ApproveAllPolicy(space), seed=0)
    assert result.capital_deployed <= budget + 1e-6


def test_binding_budget_reduces_deployment(pool, cashflow, space):
    free = make_env(pool, cashflow, space).run(ApproveAllPolicy(space), seed=0)
    tight = make_env(pool, cashflow, space, budget=100_000.0).run(ApproveAllPolicy(space), seed=0)
    assert tight.capital_deployed < free.capital_deployed
    assert tight.budget_exhausted_at is not None


def test_episode_accounting_is_consistent(pool, cashflow, space):
    result = make_env(pool, cashflow, space).run(ApproveAllPolicy(space), seed=0)
    assert result.total_reward == pytest.approx(
        result.revenue - result.losses - result.costs, rel=1e-6
    )
    assert 0.0 <= result.approval_rate <= 1.0
    assert 0.0 <= result.default_rate_of_book <= 1.0


def test_step_after_episode_end_raises(pool, cashflow, space):
    env = make_env(pool, cashflow, space, n_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        env.step(space.reject_index)
    with pytest.raises(RuntimeError, match="finished episode"):
        env.step(space.reject_index)


def test_larger_exposure_raises_pd(pool, cashflow, space):
    """The exposure-elasticity fallback must move PD in the right direction."""
    env = make_env(pool, cashflow, space)
    env.reset(seed=0)
    applicant = int(env._order[0])
    base = env._pd_at(applicant, float(pool.observed_exposure[applicant]))
    larger = env._pd_at(applicant, float(pool.observed_exposure[applicant]) * 1.5)
    assert larger > base


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def test_evaluation_is_reproducible(pool, cashflow, space):
    env = make_env(pool, cashflow, space)
    policy = FixedThresholdPolicy(0.15, space)
    first = evaluate_policy(env, policy, seeds=[0, 1, 2]).rewards()
    second = evaluate_policy(env, policy, seeds=[0, 1, 2]).rewards()
    assert np.allclose(first, second)


def test_paired_difference_against_self_is_zero(pool, cashflow, space):
    env = make_env(pool, cashflow, space)
    evals = compare_policies(env, [FixedThresholdPolicy(0.15, space)], seeds=[0, 1, 2, 3])
    name = next(iter(evals))
    result = paired_difference({**evals, "copy": evals[name]}, "copy", name)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert not result["significant"]


def test_summary_and_difference_tables_are_well_formed(pool, cashflow, space, cashflow_model=None):
    env = make_env(pool, cashflow, space)
    policies = [
        ApproveAllPolicy(space),
        FixedThresholdPolicy(0.15, space),
        ExpectedProfitPolicy(cashflow, EconomicsConfig(), space),
    ]
    evals = compare_policies(env, policies, seeds=[0, 1, 2, 3])
    summary = summary_table(evals)
    assert len(summary) == 3
    assert summary["mean_reward"].is_monotonic_decreasing
    diffs = difference_table(evals, "fixed_threshold@0.150")
    assert len(diffs) == 2
    assert {"mean_difference", "ci_low", "ci_high", "significant"} <= set(diffs.columns)
    assert (diffs["ci_low"] <= diffs["mean_difference"]).all()
    assert (diffs["mean_difference"] <= diffs["ci_high"]).all()


# ---------------------------------------------------------------------------
# RL formulation
# ---------------------------------------------------------------------------

def test_shadow_price_is_zero_when_budget_does_not_bind():
    densities = np.array([0.1, 0.08, 0.05])
    exposures = np.array([100.0, 100.0, 100.0])
    assert analytical_shadow_price(densities, exposures, budget=1_000.0) == 0.0


def test_shadow_price_is_the_marginal_applicant():
    densities = np.array([0.10, 0.08, 0.05, 0.01])
    exposures = np.array([100.0, 100.0, 100.0, 100.0])
    # Budget fits two loans; the marginal (third) applicant sets the price.
    assert analytical_shadow_price(densities, exposures, budget=250.0) == pytest.approx(0.05)


def test_shadow_price_validation_split_separation(pool, cashflow, space):
    """Test candidate analytical policy shadow price is fit on validation data, not test evaluation pool."""
    from credit_risk.decision.policies import ExpectedProfitPolicy

    # Simulate a validation pool of applicants disjoint from test pool
    val_densities = np.array([0.12, 0.09, 0.06, 0.03, 0.01])
    val_exposures = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    val_budget = 2500.0

    shadow_val = analytical_shadow_price(val_densities, val_exposures, val_budget)
    assert shadow_val > 0.0

    # Policy uses validation-fitted shadow price
    policy_val = ExpectedProfitPolicy(cashflow, EconomicsConfig(), space, capital_shadow_price=shadow_val)
    policy_val.name = f"expected_profit(lambda_val={shadow_val:.4f})"
    assert policy_val.capital_shadow_price == shadow_val
    assert "lambda_val" in policy_val.name


def test_shadow_price_rises_as_budget_tightens():
    rng = np.random.default_rng(0)
    densities = np.sort(rng.uniform(-0.05, 0.15, 200))[::-1]
    exposures = np.full(200, 1000.0)
    prices = [analytical_shadow_price(densities, exposures, b) for b in (150_000, 80_000, 30_000)]
    assert prices[0] <= prices[1] <= prices[2]


def test_state_encoder_is_deterministic(pool, cashflow, space):
    from credit_risk.decision.policies import DecisionContext

    encoder = StateEncoder(cashflow, EconomicsConfig(), space, total_budget=1e6, total_steps=100)
    context = DecisionContext(
        pool.pd_estimate[:50], pool.requested_amount[:50], pool.apr[:50],
        remaining_budget=5e5, remaining_steps=50,
    )
    assert np.array_equal(encoder.encode(context), encoder.encode(context))
    assert encoder.context_vector(context).shape == (50, encoder.context_dim)


def test_state_indices_are_in_range(pool, cashflow, space):
    from credit_risk.decision.policies import DecisionContext

    encoder = StateEncoder(cashflow, EconomicsConfig(), space, total_budget=1e6, total_steps=100)
    context = DecisionContext(pool.pd_estimate, pool.requested_amount, pool.apr,
                              remaining_budget=3e5, remaining_steps=40)
    states = encoder.encode(context)
    assert states.min() >= 0
    assert states.max() < encoder.spec.n_states


# ---------------------------------------------------------------------------
# Learning policies
# ---------------------------------------------------------------------------

def test_linucb_updates_its_posterior(pool, cashflow, space):
    encoder = StateEncoder(cashflow, EconomicsConfig(), space)
    policy = LinUCBPolicy(encoder, space, BanditConfig())
    before = policy.theta().copy()
    env = make_env(pool, cashflow, space, n_steps=200)
    env.run(policy, seed=0)
    assert policy.counts.sum() == 200
    assert not np.allclose(before, policy.theta())


def test_thompson_explores_more_than_one_action(pool, cashflow, space):
    encoder = StateEncoder(cashflow, EconomicsConfig(), space)
    policy = ThompsonSamplingPolicy(encoder, space, BanditConfig(), seed=1)
    make_env(pool, cashflow, space, n_steps=300).run(policy, seed=0)
    assert (policy.counts > 0).sum() >= 2


def test_q_learning_visits_states_and_improves(pool, cashflow, space):
    encoder = StateEncoder(cashflow, EconomicsConfig(), space, total_budget=3e5, total_steps=300)
    policy = BudgetAwareQPolicy(
        encoder, space, QLearningConfig(epsilon_decay_episodes=20, seed=0)
    )
    env = make_env(pool, cashflow, space, budget=3e5, n_steps=300)
    report = train_policy(env, policy, n_episodes=40)
    assert policy.visits.sum() > 0
    assert (policy.Q != 0).any()
    summary = report.to_dict()
    assert summary["n_episodes"] == 40
    # Learning should not make things worse over the run.
    assert summary["last_10_mean"] >= summary["first_10_mean"] * 0.7


def test_epsilon_decays_and_greedy_switches_it_off(pool, cashflow, space):
    encoder = StateEncoder(cashflow, EconomicsConfig(), space, total_budget=3e5, total_steps=100)
    policy = BudgetAwareQPolicy(
        encoder, space, QLearningConfig(epsilon_start=0.5, epsilon_end=0.05,
                                        epsilon_decay_episodes=10, seed=0)
    )
    start = policy.epsilon
    env = make_env(pool, cashflow, space, budget=3e5, n_steps=100)
    train_policy(env, policy, n_episodes=15)
    assert policy.epsilon < start
    policy.greedy()
    assert policy.epsilon == 0.0


def test_q_policy_reports_an_implied_threshold(pool, cashflow, space):
    encoder = StateEncoder(cashflow, EconomicsConfig(), space, total_budget=3e5, total_steps=200)
    policy = BudgetAwareQPolicy(encoder, space, QLearningConfig(seed=0))
    env = make_env(pool, cashflow, space, budget=3e5, n_steps=200)
    train_policy(env, policy, n_episodes=25)
    implied = policy.implied_threshold()
    assert implied["n_visited_states"] > 0
    assert len(implied["table"]) == encoder.spec.n_budget_bins * encoder.spec.n_time_bins


def test_simulator_uses_observed_outcomes_at_observed_exposure(pool, cashflow, space):
    """P1-6d: The simulator must use ground-truth observed loan outcomes and cashflows
    at observed exposure, eliminating circular coupling where the tested PD model simulates
    its own default labels."""
    cfg = SimulationConfig(
        n_steps=50,
        use_observed_outcomes_at_observed_exposure=True,
        seed=42,
    )
    env = LendingEnvironment(pool, cashflow, EconomicsConfig(), space, cfg)
    res = env.run(ApproveAllPolicy(), keep_log=True)
    assert res.per_step is not None
    for _, row in res.per_step.iterrows():
        if row["exposure"] > 0:
            applicant = int(row["applicant"])
            expected_default = bool(pool.observed_default[applicant] > 0.5)
            assert bool(row["defaulted"]) == expected_default


# ---------------------------------------------------------------------------
# CR-01: evaluation episodes must not train the policy on test-pool outcomes
# ---------------------------------------------------------------------------

def _learner_state(policy) -> bytes:
    """Byte-exact snapshot of everything a learning policy has learned."""
    arrays = [getattr(policy, name) for name in ("A", "A_inv", "b", "counts", "Q", "visits")
              if hasattr(policy, name)]
    assert arrays, "policy exposes no learnable state"
    return b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)


def _pretrained_learners(pool, cashflow, space):
    encoder = StateEncoder(cashflow, EconomicsConfig(), space, total_budget=3e5, total_steps=200)
    train_env = make_env(pool, cashflow, space, budget=3e5, n_steps=200)
    learners = [
        LinUCBPolicy(encoder, space, BanditConfig()),
        ThompsonSamplingPolicy(encoder, space, BanditConfig(), seed=1),
        BudgetAwareQPolicy(encoder, space, QLearningConfig(epsilon_decay_episodes=5, seed=0)),
    ]
    for learner in learners:
        train_policy(train_env, learner, n_episodes=3, seed_offset=10_000)
    learners[2].greedy()
    return learners


def test_evaluation_leaves_learning_policy_parameters_byte_identical(pool, cashflow, space):
    eval_env = make_env(pool, cashflow, space, budget=3e5, n_steps=200)
    for policy in _pretrained_learners(pool, cashflow, space):
        before = _learner_state(policy)
        evaluate_policy(eval_env, policy, seeds=[0, 1, 2])
        assert _learner_state(policy) == before, f"{policy.name} learned during evaluation"


def test_evaluation_results_do_not_depend_on_episode_order(pool, cashflow, space):
    eval_env = make_env(pool, cashflow, space, budget=3e5, n_steps=200)
    for policy in _pretrained_learners(pool, cashflow, space):
        forward = evaluate_policy(eval_env, policy, seeds=[0, 1, 2, 3]).rewards()
        backward = evaluate_policy(eval_env, policy, seeds=[3, 2, 1, 0]).rewards()
        alone = evaluate_policy(eval_env, policy, seeds=[2]).rewards()
        assert np.allclose(forward, backward[::-1]), policy.name
        assert np.isclose(forward[2], alone[0]), policy.name


def test_run_can_still_learn_when_asked(pool, cashflow, space):
    """Training episodes (the validation pool) must keep updating the policy."""
    encoder = StateEncoder(cashflow, EconomicsConfig(), space)
    policy = LinUCBPolicy(encoder, space, BanditConfig())
    before = _learner_state(policy)
    make_env(pool, cashflow, space, n_steps=50).run(policy, seed=0)
    assert _learner_state(policy) != before


# ---------------------------------------------------------------------------
# CR-04: headline comparisons use observed outcomes only
# ---------------------------------------------------------------------------

def test_headline_action_space_has_no_multipliers_other_than_one():
    from scripts.experiments.train_policy import build_action_space

    headline = build_action_space(40_000.0, include_review=False, multi_exposure=False)
    assert set(headline.multipliers.tolist()) == {0.0, 1.0}
    assert not any(a.is_review for a in headline.actions)
    sensitivity = build_action_space(40_000.0, include_review=False, multi_exposure=True)
    assert {0.5, 1.5} <= set(sensitivity.multipliers.tolist())


def test_accept_reject_episodes_earn_only_realised_outcomes(pool, cashflow, space):
    headline = space.at_observed_exposure()
    assert pool.observed_exposure.tolist() == pool.requested_amount.tolist(), "fixture must fund as requested"
    env = make_env(pool, cashflow, headline, n_steps=300)
    policies = [
        FixedThresholdPolicy(0.15, headline),
        ExpectedProfitPolicy(cashflow, EconomicsConfig(), headline),
        RiskBandPolicy(action_space=headline),
    ]
    for policy in policies:
        result = env.run(policy, seed=3, keep_log=True, learn=False)
        log = result.per_step
        approved = log[log["exposure"] > 0]
        assert len(approved) > 0, policy.name
        idx = approved["applicant"].to_numpy()
        # every approval is at the observed exposure and reproduces the observed outcome
        assert np.allclose(approved["exposure"].to_numpy(), pool.observed_exposure[idx])
        assert (approved["defaulted"].to_numpy() == (pool.observed_default[idx] > 0.5)).all()


def test_verdict_is_not_sign_blind():
    """G4: a significantly *negative* best difference must not be reported as a win."""
    from credit_risk.simulation.evaluation import verdict

    diff = pd.DataFrame(
        {"mean_difference": [-28_262.0], "ci_low": [-54_837.0], "ci_high": [-2_616.0],
         "significant": [True], "relative_improvement": [-0.4], "win_rate": [0.2]},
        index=["expected_profit"],
    )
    text = verdict(diff, "fixed_threshold@0.150")
    assert "beat fixed_threshold" not in text.replace("No policy beat fixed_threshold", "")
    assert "significantly worse" in text

    diff["mean_difference"], diff["ci_low"], diff["ci_high"] = 5_000.0, 1_000.0, 9_000.0
    assert "beat fixed_threshold@0.150 by +5,000" in verdict(diff, "fixed_threshold@0.150")
