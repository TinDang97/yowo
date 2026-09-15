"""The decode tail must be COMPUTED from the graph, never recited from a list.

The shipping INT8 export quantized every node, including the post-head decode
subgraph (DFL softmax, box decode, final sigmoid). Measured 2026-09-15 on a
production-path ``yolo11n`` FP32 ONNX export: the quantized artifact returns a
well-formed ``(1, 84, 8400)`` tensor whose 4 box rows still decode but whose
80 class rows are every one exactly 0.0 — five
detections become zero.

The rule that fixes it is topological: walk backward from every graph output
and stop at the first ``Conv`` on each path. Everything reached before a Conv
is the decode tail and is excluded from quantization; the Convs themselves —
where the whole size win lives — still quantize.

These checks exist to stop that rule from decaying into a hardcoded node list.
Every graph here is synthetic and built inline, so a perturbation is a two-line
edit and the assertion is about the RULE, not about ``yolo11n``.
"""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from yowo.export._int8 import decode_tail  # noqa: E402

# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _val(name: str) -> object:
    """A float tensor value-info with an unspecified shape."""
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)


def _model(nodes: list[object], outputs: list[str]) -> object:
    """Wrap ``nodes`` in a ModelProto with ``images`` in and ``outputs`` out."""
    graph = helper.make_graph(
        nodes,  # type: ignore[arg-type]
        "g",
        [_val("images")],  # type: ignore[list-item]
        [_val(o) for o in outputs],  # type: ignore[misc]
    )
    return helper.make_model(graph)


def _linear_graph() -> object:
    """images -> Conv(backbone) -> Conv(head) -> Reshape -> Softmax -> Sigmoid -> out.

    Tail is the three post-head ops; both Convs and nothing above them.
    """
    return _model(
        [
            helper.make_node("Conv", ["images"], ["b"], name="backbone"),
            helper.make_node("Conv", ["b"], ["h"], name="head"),
            helper.make_node("Reshape", ["h"], ["r"], name="reshape"),
            helper.make_node("Softmax", ["r"], ["s"], name="softmax"),
            helper.make_node("Sigmoid", ["s"], ["out"], name="sigmoid"),
        ],
        ["out"],
    )


# ---------------------------------------------------------------------------
# The rule is computed, not recited
# ---------------------------------------------------------------------------


def test_decode_tail_stops_at_conv() -> None:
    """The terminating Conv, and everything above it, stays quantizable."""
    tail = decode_tail(_linear_graph())
    assert tail == {"reshape", "softmax", "sigmoid"}
    assert "head" not in tail, "the head Conv must stay quantizable — that is the size win"
    assert "backbone" not in tail


def test_decode_tail_follows_a_perturbed_graph() -> None:
    """Insert one op into the decode region; the tail grows by exactly that node.

    A hardcoded list, an op-type allowlist, or a node-index heuristic all fail
    here: the new node is named nothing the implementation could have known.
    """
    before = decode_tail(_linear_graph())

    perturbed = _model(
        [
            helper.make_node("Conv", ["images"], ["b"], name="backbone"),
            helper.make_node("Conv", ["b"], ["h"], name="head"),
            helper.make_node("Reshape", ["h"], ["r"], name="reshape"),
            # a new op nobody wrote a rule for, in the middle of the tail
            helper.make_node("Erf", ["r"], ["e"], name="a_node_no_rule_names"),
            helper.make_node("Softmax", ["e"], ["s"], name="softmax"),
            helper.make_node("Sigmoid", ["s"], ["out"], name="sigmoid"),
        ],
        ["out"],
    )
    after = decode_tail(perturbed)

    assert after - before == {"a_node_no_rule_names"}
    assert after == before | {"a_node_no_rule_names"}


def test_decode_tail_follows_a_shortened_graph() -> None:
    """Remove an op from the decode region; the tail shrinks by exactly that node."""
    before = decode_tail(_linear_graph())

    shortened = _model(
        [
            helper.make_node("Conv", ["images"], ["b"], name="backbone"),
            helper.make_node("Conv", ["b"], ["h"], name="head"),
            helper.make_node("Reshape", ["h"], ["r"], name="reshape"),
            helper.make_node("Sigmoid", ["r"], ["out"], name="sigmoid"),
        ],
        ["out"],
    )
    assert before - decode_tail(shortened) == {"softmax"}


def test_decode_tail_crosses_a_diamond_once() -> None:
    """Conv-free reachability on ANY path puts a node in; the set holds it once.

    ``shared`` is reachable from the output without crossing a Conv (via
    ``left``) and also sits behind one (via the Conv ``right``). The union
    reading puts it in. ``behind_conv`` is reachable ONLY through a Conv and
    must stay out.
    """
    model = _model(
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Mul", ["h"], ["shared"], name="shared"),
            helper.make_node("Mul", ["shared"], ["z"], name="behind_conv"),
            helper.make_node("Conv", ["z"], ["q"], name="right"),
            helper.make_node("Mul", ["shared"], ["p"], name="left"),
            helper.make_node("Add", ["p", "q"], ["out"], name="join"),
        ],
        ["out"],
    )
    tail = decode_tail(model)
    assert tail == {"join", "left", "shared"}
    assert "behind_conv" not in tail, "only reachable through a Conv — stays quantizable"
    assert "right" not in tail
    assert len(tail) == 3, "a set, visited once, regardless of walk order"


def test_decode_tail_covers_every_graph_output() -> None:
    """A second output is walked too, not just the first."""
    model = _model(
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Sigmoid", ["h"], ["out0"], name="branch_a"),
            helper.make_node("Softmax", ["h"], ["out1"], name="branch_b"),
        ],
        ["out0", "out1"],
    )
    assert decode_tail(model) == {"branch_a", "branch_b"}


def test_decode_tail_names_not_indices() -> None:
    """Every element is a node NAME that ``nodes_to_exclude`` can address."""
    model = _linear_graph()
    names = {n.name for n in model.graph.node}  # type: ignore[attr-defined]
    tail = decode_tail(model)
    assert tail <= names
    assert all(isinstance(t, str) for t in tail)


def test_decode_tail_is_empty_when_the_output_is_a_conv() -> None:
    """A graph whose output is produced by a Conv has no decode tail at all."""
    model = _model([helper.make_node("Conv", ["images"], ["out"], name="head")], ["out"])
    assert decode_tail(model) == set()


def test_decode_tail_halts_on_a_cycle() -> None:
    """A malformed cyclic graph terminates rather than spinning forever."""
    model = _model(
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Add", ["h", "b"], ["a"], name="loop_a"),
            helper.make_node("Mul", ["a"], ["b"], name="loop_b"),
            helper.make_node("Sigmoid", ["a"], ["out"], name="sigmoid"),
        ],
        ["out"],
    )
    assert decode_tail(model) == {"sigmoid", "loop_a", "loop_b"}


# ---------------------------------------------------------------------------
# An unaddressable node is a refusal, not a silent omission
# ---------------------------------------------------------------------------


def test_decode_tail_rejects_an_unnamed_tail_node() -> None:
    """``nodes_to_exclude`` matches by name; an unnamed tail node cannot be excluded.

    Returning the set minus that node would under-exclude silently and
    reproduce the exact defect this module exists to kill, so it raises.
    """
    from yowo.errors import ExportError

    model = _model(
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Softmax", ["h"], ["s"], name=""),
            helper.make_node("Sigmoid", ["s"], ["out"], name="sigmoid"),
        ],
        ["out"],
    )
    with pytest.raises(ExportError, match="name"):
        decode_tail(model)


def test_decode_tail_keeps_both_nodes_that_share_a_name() -> None:
    """ONNX does not enforce unique node names, and a name-keyed walk misses one.

    This is the real reason the traversal is keyed on TENSORS. A name-keyed
    ``seen`` set would still HALT on a cycle — it drains the stack either way —
    so ``test_decode_tail_halts_on_a_cycle`` does NOT justify the choice, and
    nothing else did until this check.
    """
    model = _model(
        [
            helper.make_node("Conv", ["images"], ["h"], name="head"),
            helper.make_node("Mul", ["h"], ["a"], name="twin"),
            helper.make_node("Softmax", ["a"], ["b"], name="twin"),
            helper.make_node("Sigmoid", ["b"], ["out"], name="sigmoid"),
        ],
        ["out"],
    )
    producer = {o: n.name for n in model.graph.node for o in n.output}
    assert producer["a"] == producer["b"] == "twin", "the graph really does reuse a name"

    # A name-keyed walk records the Softmax `twin`, then refuses to visit the
    # Mul `twin` because the name is already seen — so it never enqueues `h`,
    # never learns the walk terminates at a Conv, and under-excludes the Mul.
    assert decode_tail(model) == {"sigmoid", "twin"}
    assert "head" not in decode_tail(model)


def test_an_unnamed_node_outside_the_tail_is_tolerated() -> None:
    """The refusal is about what must be EXCLUDED, not about graph hygiene."""
    model = _model(
        [
            helper.make_node("Conv", ["images"], ["b"], name=""),
            helper.make_node("Conv", ["b"], ["h"], name="head"),
            helper.make_node("Sigmoid", ["h"], ["out"], name="sigmoid"),
        ],
        ["out"],
    )
    assert decode_tail(model) == {"sigmoid"}
