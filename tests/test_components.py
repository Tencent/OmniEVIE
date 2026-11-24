"""CPU unit tests."""

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from omnievie import DEFAULT_MODEL_ID, cli, localize_image, localize_video, media, pooling
from omnievie.gate import NUM_FEATURES, QueryGate, query_features
from omnievie.index import NOISE_BETA, NOISE_THETA, SCALE, ThreeStageIndex


def _write_gate(path):
    torch.manual_seed(7)
    tensors = {
        "mlp_w1": torch.randn(16, NUM_FEATURES) * 0.1,
        "mlp_b1": torch.zeros(16),
        "mlp_w2": torch.randn(1, 16) * 0.1,
        "mlp_b2": torch.zeros(1),
        "dir_w": torch.randn(1, 64) * 0.1,
    }
    save_file(tensors, str(path), metadata={"kind": "salience_v1", "tau": "0.8"})
    return path


def _unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def _brute_force(query, docs, weights):
    """Exact stage-3 score: int8 MaxSim, noise-floor transform, weighted sum over query tokens."""
    scores = []
    for doc in docs:
        codes = torch.round(doc * SCALE).clamp(-127, 127) / SCALE
        best = (query @ codes.T).amax(dim=1)
        boosted = best + NOISE_BETA * F.relu(best - NOISE_THETA)
        scores.append(float((boosted * weights).sum()))
    return torch.tensor(scores)


def test_index_matches_brute_force_and_round_trips(tmp_path):
    torch.manual_seed(0)
    docs = [_unit(int(torch.randint(1, 40, (1,))), 64) for _ in range(120)]
    index = ThreeStageIndex(n1=len(docs), n2=len(docs))
    for i, doc in enumerate(docs):
        index.add(f"doc{i}", doc, "text")

    query = _unit(12, 64)
    weights = torch.rand(12) + 0.5
    weights = weights * (12 / weights.sum())
    expected = _brute_force(query, docs, weights)
    top = torch.topk(expected, 5)

    got = index.search(query, weights=weights, modality="text", topk=5)
    assert [name for name, _ in got] == [f"doc{int(i)}" for i in top.indices]
    assert all(abs(score - float(top.values[j])) < 1e-3 for j, (_, score) in enumerate(got))

    path = tmp_path / "index.pt"
    index.save(path)
    reloaded = ThreeStageIndex.load(path)
    assert reloaded.search(query, weights=weights, modality="text", topk=5) == got


def test_index_modality_filter():
    torch.manual_seed(1)
    index = ThreeStageIndex()
    index.add("t", _unit(5, 64), "text")
    index.add("v", _unit(5, 64), "video")
    assert [n for n, _ in index.search(_unit(4, 64), modality="video", topk=5)] == ["v"]


def test_candidates_restrict_the_pool():
    torch.manual_seed(4)
    index = ThreeStageIndex()
    for i in range(10):
        index.add(f"d{i}", _unit(6, 64), "image")
    got = index.search(_unit(5, 64), candidates=["d2", "d7"], topk=5)
    assert sorted(name for name, _ in got) == ["d2", "d7"]


def test_gate_weights_sum_to_length_and_features_are_shaped(tmp_path):
    gate_file = _write_gate(tmp_path / "query_gate.safetensors")
    gate = QueryGate.load(gate_file)
    gate_from_dir = QueryGate.from_pretrained(tmp_path)
    gate_from_file = QueryGate.from_pretrained(gate_file)
    torch.manual_seed(2)
    raw = torch.randn(20, 64)
    vision = torch.zeros(20, dtype=torch.bool)
    vision[3:9] = True
    aug = torch.zeros(20, dtype=torch.bool)
    aug[-10:] = True
    feats = query_features(raw, F.normalize(raw, dim=-1), vision, aug, has_text=True)
    assert feats.shape == (20, NUM_FEATURES)
    weights = gate.weights(raw, vision, aug, has_text=True)
    assert weights.shape == (20,)
    assert abs(float(weights.sum()) - 20.0) < 1e-4
    assert float(weights.min()) > 0.0
    assert torch.allclose(weights, gate_from_dir.weights(raw, vision, aug, has_text=True))
    assert torch.allclose(weights, gate_from_file.weights(raw, vision, aug, has_text=True))


def test_pooling_respects_budget_and_unit_norm():
    torch.manual_seed(3)
    tokens = torch.randn(300, 64)                 # 40 x 30 patches merged 2x2 -> 20 x 15 tokens
    image = pooling.image(tokens, [1, 40, 30])
    assert image.shape[0] == pooling.BUDGET
    assert torch.allclose(image.norm(dim=-1), torch.ones(image.shape[0]), atol=1e-5)
    short = pooling.text_window(torch.randn(7, 64))
    assert short.shape == (7, 64)
    merged = pooling.merge([torch.randn(100, 64), torch.randn(100, 64)], budget=64)
    assert merged.shape[0] == 64


def test_mmeb_frame_rule_matches_loader():
    # 64 saved frames are spread over the video; 32 of them are read, evenly spaced, in order.
    assert media.saved_frame_indices(100)[:3] == [0, 1, 3]
    assert media.saved_frame_indices(30) == list(range(30))
    picked = media.uniform_frames(list(range(64)), media.MMEB_FRAMES)
    assert len(picked) == 32 and picked[0] == 0 and picked[-1] == 63 and picked == sorted(picked)
    assert media.uniform_frames(list(range(20)), 32) == list(range(20))   # fewer frames: keep all
    assert media.protocol_settings("mmeb")["token_cap"] == 14400


def test_localize_image_and_video():
    torch.manual_seed(11)
    # 10 x 10 merged grid (grid_thw = [1, 20, 20]): plant a small target at (row=9, col=9) (1% area)
    # and a uniform background cluster across the rest (>5% area) to verify background filtering.
    target = _unit(64)
    bg = _unit(64)
    bg = F.normalize(bg - (bg @ target) * target, dim=-1)
    tokens = bg.unsqueeze(0).repeat(100, 1)
    tokens[99] = target
    img_loc = localize_image(torch.stack([target, bg]), tokens, [1, 20, 20], budget=4)
    assert img_loc.grid_hw == (10, 10)
    assert img_loc.top_box is not None
    assert img_loc.top_box[0] >= 0.9 and img_loc.top_box[1] >= 0.9
    assert float(img_loc.heat_map[9, 9]) > 0.0
    assert len(img_loc.token_hits) == 1  # large background cluster skipped by default max_area_ratio=0.05

    # Video with 4 temporal groups over 120s: plant a strong match in group 2 (60-90s)
    grids = [[1, 4, 4]] * 4  # 4 tokens per group, 16 tokens total
    v_tokens = _unit(16, 64) * 0.05
    v_tokens[8:12] = target
    v_tokens = F.normalize(v_tokens, dim=-1)
    vid_loc = localize_video(target.unsqueeze(0), [(v_tokens, grids)], duration=120.0, budget=16)
    assert 60.0 <= vid_loc.peak_sec <= 90.0
    assert vid_loc.span_sec[0] <= vid_loc.peak_sec <= vid_loc.span_sec[1]


def test_cli_parser_defaults():
    args = cli.build_parser().parse_args(["search", "idx.pt", "--text", "hello"])
    assert args.weights == DEFAULT_MODEL_ID
    assert args.target == "text"

