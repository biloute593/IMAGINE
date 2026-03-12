"""Tests for SceneGraph light module."""

import numpy as np

from photo_editor.scene_graph import SceneGraph, SceneNode, SceneEdge


class TestSceneGraphNodeManagement:
    def test_add_node(self):
        sg = SceneGraph()
        mask = np.zeros((10, 10), dtype=bool)
        sg.add_node("obj1", mask, (0, 0, 5, 5), depth_mean=0.5)
        assert "obj1" in sg.nodes
        assert sg.nodes["obj1"].depth_mean == 0.5

    def test_remove_node(self):
        sg = SceneGraph()
        mask = np.zeros((10, 10), dtype=bool)
        sg.add_node("obj1", mask, (0, 0, 5, 5))
        sg.remove_node("obj1")
        assert "obj1" not in sg.nodes

    def test_remove_node_removes_edges(self):
        sg = SceneGraph()
        mask = np.zeros((10, 10), dtype=bool)
        sg.add_node("a", mask, (0, 0, 5, 5))
        sg.add_node("b", mask, (0, 0, 5, 5))
        sg.edges.append(SceneEdge("a", "b", "adjacent"))
        sg.remove_node("a")
        assert len(sg.edges) == 0


class TestSceneGraphRelations:
    def test_adjacent_detection(self):
        sg = SceneGraph()
        mask1 = np.zeros((100, 100), dtype=bool)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask1[10:20, 10:20] = True
        mask2[10:20, 30:40] = True
        sg.add_node("a", mask1, (10, 10, 10, 10), depth_mean=0.5)
        sg.add_node("b", mask2, (30, 10, 10, 10), depth_mean=0.5)
        edges = sg.build_edges()
        relations = [e.relation for e in edges]
        assert "adjacent" in relations

    def test_touches_detection(self):
        sg = SceneGraph()
        mask1 = np.zeros((50, 50), dtype=bool)
        mask2 = np.zeros((50, 50), dtype=bool)
        mask1[10:20, 10:20] = True
        mask2[10:20, 20:30] = True  # directly adjacent
        sg.add_node("a", mask1, (10, 10, 10, 10))
        sg.add_node("b", mask2, (20, 10, 10, 10))
        edges = sg.build_edges()
        relations = [e.relation for e in edges]
        assert "touches" in relations

    def test_occlusion_detection(self):
        sg = SceneGraph()
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True  # same region = overlap
        sg.add_node("front", mask.copy(), (10, 10, 10, 10), depth_mean=0.2)
        sg.add_node("back", mask.copy(), (10, 10, 10, 10), depth_mean=0.8)
        edges = sg.build_edges()
        occlusion = [e for e in edges if e.relation == "occludes"]
        assert len(occlusion) > 0
        assert occlusion[0].source == "front"

    def test_no_self_relations(self):
        sg = SceneGraph()
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        sg.add_node("a", mask, (2, 2, 3, 3))
        edges = sg.build_edges()
        for e in edges:
            assert e.source != e.target


class TestSceneGraphPropagation:
    def test_propagate_changes(self):
        sg = SceneGraph()
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        sg.add_node("a", mask, (10, 10, 10, 10))
        sg.add_node("b", mask, (10, 10, 10, 10))
        sg.edges.append(SceneEdge("a", "b", "adjacent"))
        impacted = sg.propagate_changes("a")
        assert "b" in impacted

    def test_propagate_no_impact(self):
        sg = SceneGraph()
        mask = np.zeros((50, 50), dtype=bool)
        sg.add_node("a", mask, (0, 0, 5, 5))
        sg.add_node("b", mask, (40, 40, 5, 5))
        impacted = sg.propagate_changes("a")
        assert "b" not in impacted

    def test_get_relations(self):
        sg = SceneGraph()
        mask = np.zeros((10, 10), dtype=bool)
        sg.add_node("a", mask, (0, 0, 5, 5))
        sg.add_node("b", mask, (0, 0, 5, 5))
        sg.edges.append(SceneEdge("a", "b", "touches"))
        rels = sg.get_relations("a")
        assert len(rels) == 1
        assert rels[0].relation == "touches"
