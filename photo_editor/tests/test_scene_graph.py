"""
test_scene_graph.py — Tests unitaires pour le module SceneGraph
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from photo_editor.scene_graph import SceneGraph, SceneNode, SceneEdge


def _make_mask(h=100, w=100, region=None):
    mask = np.zeros((h, w), dtype=bool)
    if region:
        y1, y2, x1, x2 = region
        mask[y1:y2, x1:x2] = True
    return mask


class TestAddNode:
    def test_add_single_node(self):
        sg = SceneGraph()
        mask = _make_mask(region=(10, 30, 10, 30))
        node = sg.add_node("obj_1", mask, (10, 10, 30, 30), depth_mean=0.3)
        assert "obj_1" in sg.nodes
        assert isinstance(node, SceneNode)

    def test_node_has_correct_attributes(self):
        sg = SceneGraph()
        mask = _make_mask(region=(5, 15, 5, 15))
        sg.add_node("obj_a", mask, (5, 5, 15, 15), depth_mean=0.4,
                    signature={"material_class": "metal_speculaire"})
        n = sg.get_node("obj_a")
        assert n.id == "obj_a"
        assert n.depth_mean == 0.4
        assert n.signature["material_class"] == "metal_speculaire"

    def test_get_nonexistent_node(self):
        sg = SceneGraph()
        assert sg.get_node("missing") is None


class TestRemoveNode:
    def test_remove_node_and_edges(self):
        sg = SceneGraph()
        m1 = _make_mask(region=(10, 20, 10, 20))
        m2 = _make_mask(region=(12, 22, 12, 22))
        sg.add_node("a", m1, (10, 10, 20, 20), depth_mean=0.3)
        sg.add_node("b", m2, (12, 12, 22, 22), depth_mean=0.5)
        sg.remove_node("a")
        assert "a" not in sg.nodes
        for e in sg.edges:
            assert e.source != "a" and e.target != "a"


class TestRelations:
    def test_adjacent_relation(self):
        sg = SceneGraph()
        m1 = _make_mask(region=(10, 20, 10, 20))
        m2 = _make_mask(region=(10, 20, 50, 60))
        sg.add_node("a", m1, (10, 10, 20, 20), depth_mean=0.5)
        sg.add_node("b", m2, (10, 50, 20, 60), depth_mean=0.5)
        edges = sg.get_relations("a", "b")
        relations = [e.relation for e in edges]
        assert "adjacent" in relations

    def test_occludes_relation(self):
        sg = SceneGraph()
        m1 = _make_mask(region=(10, 20, 10, 20))
        m2 = _make_mask(region=(10, 20, 10, 20))
        sg.add_node("front", m1, (10, 10, 20, 20), depth_mean=0.1)
        sg.add_node("back", m2, (10, 10, 20, 20), depth_mean=0.9)
        edges = sg.get_relations("front", "back")
        occludes = [e for e in edges if e.relation == "occludes"]
        assert len(occludes) > 0
        assert occludes[0].source == "front"

    def test_no_duplicate_edges(self):
        sg = SceneGraph()
        m1 = _make_mask(region=(5, 15, 5, 15))
        m2 = _make_mask(region=(5, 15, 40, 50))
        sg.add_node("x", m1, (5, 5, 15, 15), depth_mean=0.5)
        sg.add_node("y", m2, (5, 40, 15, 50), depth_mean=0.5)
        relations = [e.relation for e in sg.get_relations("x", "y")]
        assert len(relations) == len(set(relations))


class TestPropagateChanges:
    def test_propagate_returns_neighbours(self):
        sg = SceneGraph()
        m1 = _make_mask(region=(10, 20, 10, 20))
        m2 = _make_mask(region=(10, 20, 30, 40))
        sg.add_node("center", m1, (10, 10, 20, 20), depth_mean=0.5)
        sg.add_node("neighbor", m2, (10, 30, 20, 40), depth_mean=0.5)
        impacted = sg.propagate_changes("center")
        impacted_ids = [n.id for n in impacted]
        assert "neighbor" in impacted_ids

    def test_propagate_empty_graph(self):
        sg = SceneGraph()
        sg.add_node("solo", _make_mask(), (0, 0, 10, 10), depth_mean=0.5)
        assert sg.propagate_changes("solo") == []


class TestGetEdges:
    def test_get_edges_node(self):
        sg = SceneGraph()
        m1 = _make_mask(region=(10, 20, 10, 20))
        m2 = _make_mask(region=(10, 20, 50, 60))
        sg.add_node("a", m1, (10, 10, 20, 20), depth_mean=0.5)
        sg.add_node("b", m2, (10, 50, 20, 60), depth_mean=0.5)
        edges = sg.get_edges("a")
        assert len(edges) > 0
        for e in edges:
            assert e.source == "a" or e.target == "a"
