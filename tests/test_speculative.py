"""Unit tests for PagedServe Speculative Decoding Engine."""

from pagedserve.speculative import SpeculativeEngine


def test_speculative_all_accepted_greedy():
    engine = SpeculativeEngine(num_speculative_tokens=3)
    draft_tokens = [10, 20, 30]
    # Equal distributions where 10, 20, 30 are argmax
    d_p1 = [0.0] * 50; d_p1[10] = 1.0
    d_p2 = [0.0] * 50; d_p2[20] = 1.0
    d_p3 = [0.0] * 50; d_p3[30] = 1.0

    t_p1 = [0.0] * 50; t_p1[10] = 1.0
    t_p2 = [0.0] * 50; t_p2[20] = 1.0
    t_p3 = [0.0] * 50; t_p3[30] = 1.0
    t_p4 = [0.0] * 50; t_p4[40] = 1.0 # bonus token

    res = engine.verify_and_sample(
        draft_tokens=draft_tokens,
        draft_probs=[d_p1, d_p2, d_p3],
        target_probs=[t_p1, t_p2, t_p3, t_p4],
        temperature=0.0,
    )

    assert res.accepted_tokens == [10, 20, 30, 40]
    assert res.num_drafted == 3
    assert res.num_accepted == 3
    assert res.acceptance_rate == 1.0


def test_speculative_rejection_greedy():
    engine = SpeculativeEngine(num_speculative_tokens=2)
    draft_tokens = [10, 20]
    d_p1 = [0.0] * 50; d_p1[10] = 1.0
    d_p2 = [0.0] * 50; d_p2[20] = 1.0

    t_p1 = [0.0] * 50; t_p1[10] = 1.0
    t_p2 = [0.0] * 50; t_p2[25] = 1.0 # Target disagrees at token 2!

    res = engine.verify_and_sample(
        draft_tokens=draft_tokens,
        draft_probs=[d_p1, d_p2],
        target_probs=[t_p1, t_p2],
        temperature=0.0,
    )

    assert res.accepted_tokens == [10, 25] # Token 10 accepted, token 20 rejected & replaced with 25
    assert res.num_drafted == 2
    assert res.num_accepted == 1
    assert res.acceptance_rate == 0.5
