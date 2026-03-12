"""
SceneGraph Light — Relations topologiques uniquement.
Les effets physiques (ombre, reflet, lumière) sont délégués à SpatialField.

Relations gérées : occludes, touches, adjacent, supports.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class SceneNode:
    """A single object in the scene graph."""
    id: str
    mask: np.ndarray  # (H, W) bool
    bbox: tuple  # (x, y, w, h)
    depth_mean: float = 0.0
    signature: dict = field(default_factory=dict)


@dataclass
class SceneEdge:
    """Directed relation between two scene nodes."""
    source: str
    target: str
    relation: str  # occludes | touches | adjacent | supports
    weight: float = 1.0


class SceneGraph:
    """Lightweight scene graph — topology only."""

    ADJACENT_THRESHOLD = 150  # pixels

    def __init__(self):
        self.nodes: dict[str, SceneNode] = {}
        self.edges: list[SceneEdge] = []

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        mask: np.ndarray,
        bbox: tuple,
        depth_mean: float = 0.0,
        signature: dict | None = None,
    ) -> SceneNode:
        node = SceneNode(
            id=node_id,
            mask=mask.astype(bool),
            bbox=bbox,
            depth_mean=depth_mean,
            signature=signature or {},
        )
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [
            e for e in self.edges
            if e.source != node_id and e.target != node_id
        ]

    # ------------------------------------------------------------------
    # Relation detection
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_center(bbox: tuple) -> tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    def build_edges(self) -> list[SceneEdge]:
        """Compute all topological relations between nodes."""
        self.edges.clear()
        ids = list(self.nodes.keys())

        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                a = self.nodes[id_a]
                b = self.nodes[id_b]
                self._detect_relations(a, b)

        return self.edges

    def _detect_relations(self, a: SceneNode, b: SceneNode) -> None:
        # --- Occlusion (depth based) ---
        overlap = a.mask & b.mask
        if overlap.any():
            if a.depth_mean < b.depth_mean:
                self.edges.append(SceneEdge(a.id, b.id, "occludes"))
            else:
                self.edges.append(SceneEdge(b.id, a.id, "occludes"))

        # --- Touches (adjacent masks, 1px dilation) ---
        dilated_a = binary_dilation(a.mask, iterations=1)
        if (dilated_a & b.mask).any():
            self.edges.append(SceneEdge(a.id, b.id, "touches"))

        # --- Adjacent (< ADJACENT_THRESHOLD px between centers) ---
        ca = self._bbox_center(a.bbox)
        cb = self._bbox_center(b.bbox)
        dist = np.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
        if dist < self.ADJACENT_THRESHOLD:
            self.edges.append(SceneEdge(a.id, b.id, "adjacent"))

        # --- Supports (a is below b and overlapping horizontally) ---
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        h_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        if h_overlap > 0:
            if (ay + ah) >= by and a.depth_mean >= b.depth_mean:
                self.edges.append(SceneEdge(a.id, b.id, "supports"))
            elif (by + bh) >= ay and b.depth_mean >= a.depth_mean:
                self.edges.append(SceneEdge(b.id, a.id, "supports"))

    # ------------------------------------------------------------------
    # Propagation (topology only)
    # ------------------------------------------------------------------

    def propagate_changes(self, source_id: str) -> list[str]:
        """Return IDs of nodes topologically impacted by *source_id*.

        Physical impact calculation is delegated to SpatialField.
        """
        impacted: set[str] = set()
        for edge in self.edges:
            if edge.source == source_id and edge.target != source_id:
                impacted.add(edge.target)
            if edge.target == source_id and edge.source != source_id:
                impacted.add(edge.source)
        return list(impacted)

    def get_relations(self, node_id: str) -> list[SceneEdge]:
        """Return all edges involving *node_id*."""
        return [
            e for e in self.edges
            if e.source == node_id or e.target == node_id
        ]
