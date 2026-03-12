"""
scene_graph.py — SceneGraph light (topologie uniquement)
Gère les relations structurelles entre objets : occlusion, adjacence,
contact, support.

RÈGLE 6 — SPATIAL FIELD AVANT SCENE GRAPH :
  Les effets physiques (ombre, reflet, lumière) → SpatialField.
  Ce module ne gère QUE la topologie.

Relations :
  occludes   — A est devant B (depth)
  touches    — A et B ont des masques adjacents
  adjacent   — A et B sont proches (< 150px)
  supports   — A est posé sur B (depth + position relative)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ADJACENCY_THRESHOLD = 150  # pixels


@dataclass
class SceneNode:
    """Représentation d'un objet dans le graphe de scène."""

    id: str
    mask: np.ndarray
    bbox: tuple              # (x1, y1, x2, y2)
    depth_mean: float
    signature: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class SceneEdge:
    """Relation dirigée entre deux objets."""

    source: str
    target: str
    relation: str   # occludes | touches | adjacent | supports
    weight: float = 1.0


class SceneGraph:
    """
    Graphe de scène léger : topologie uniquement.
    Les effets physiques sont délégués au SpatialField.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, SceneNode] = {}
        self._edges: list[SceneEdge] = []

    # ------------------------------------------------------------------
    # Gestion des nœuds
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        mask: np.ndarray,
        bbox: tuple,
        depth_mean: float,
        signature: Optional[dict] = None,
    ) -> SceneNode:
        """
        Ajoute un objet au graphe et met à jour les relations automatiquement.

        Args:
            node_id    : identifiant unique de l'objet.
            mask       : masque binaire (H, W) bool.
            bbox       : (x1, y1, x2, y2).
            depth_mean : profondeur moyenne MiDaS [0, 1].
            signature  : signature 8D optionnelle.

        Returns:
            Le SceneNode créé.
        """
        node = SceneNode(
            id=node_id,
            mask=mask,
            bbox=bbox,
            depth_mean=depth_mean,
            signature=signature or {},
        )
        self._nodes[node_id] = node
        self._update_relations(node)
        return node

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [
            e for e in self._edges
            if e.source != node_id and e.target != node_id
        ]

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def _update_relations(self, new_node: SceneNode) -> None:
        """Calcule et ajoute les relations entre le nouveau nœud et l'existant."""
        for other_id, other_node in self._nodes.items():
            if other_id == new_node.id:
                continue
            self._compute_relations(new_node, other_node)

    def _compute_relations(self, node_a: SceneNode, node_b: SceneNode) -> None:
        """Détermine les relations topologiques entre deux nœuds."""
        # occludes : A est devant B si depth_a < depth_b (plus proche caméra)
        if node_a.depth_mean < node_b.depth_mean - 0.05:
            self._add_edge(node_a.id, node_b.id, "occludes", weight=1.0)
        elif node_b.depth_mean < node_a.depth_mean - 0.05:
            self._add_edge(node_b.id, node_a.id, "occludes", weight=1.0)

        # Distance entre centres
        cx_a, cy_a = node_a.center
        cx_b, cy_b = node_b.center
        dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

        # touches : masques adjacents (pixels voisins)
        if self._masks_touch(node_a.mask, node_b.mask):
            self._add_edge(node_a.id, node_b.id, "touches", weight=1.0)

        # adjacent : centres proches
        if dist < ADJACENCY_THRESHOLD:
            self._add_edge(node_a.id, node_b.id, "adjacent", weight=1.0 - dist / ADJACENCY_THRESHOLD)

        # supports : A est posé sur B si A est au-dessus de B en y
        # et profondeur similaire (même plan)
        ya2 = node_a.bbox[3]
        yb1 = node_b.bbox[1]
        if abs(ya2 - yb1) < 20 and abs(node_a.depth_mean - node_b.depth_mean) < 0.1:
            self._add_edge(node_b.id, node_a.id, "supports", weight=0.8)

    def _masks_touch(self, mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
        """Vérifie si deux masques binaires ont des pixels adjacents."""
        if mask_a.shape != mask_b.shape:
            return False
        from scipy.ndimage import binary_dilation
        dilated_a = binary_dilation(mask_a, iterations=2)
        return bool(np.any(dilated_a & mask_b))

    def _add_edge(
        self, source: str, target: str, relation: str, weight: float = 1.0
    ) -> None:
        # Évite les doublons
        for e in self._edges:
            if e.source == source and e.target == target and e.relation == relation:
                e.weight = weight
                return
        self._edges.append(SceneEdge(source, target, relation, weight))

    # ------------------------------------------------------------------
    # Propagation topologique
    # ------------------------------------------------------------------

    def propagate_changes(self, modified_node_id: str) -> list:
        """
        Retourne les objets topologiquement impactés par la modification.
        Le calcul physique de l'impact est délégué à SpatialField.

        Args:
            modified_node_id : identifiant de l'objet modifié.

        Returns:
            Liste d'objets (SceneNode) impactés (voisins directs).
        """
        impacted = []
        for edge in self._edges:
            other_id = None
            if edge.source == modified_node_id:
                other_id = edge.target
            elif edge.target == modified_node_id:
                other_id = edge.source

            if other_id and other_id in self._nodes:
                node = self._nodes[other_id]
                if node not in impacted:
                    impacted.append(node)
        return impacted

    # ------------------------------------------------------------------
    # Accesseurs
    # ------------------------------------------------------------------

    def get_edges(self, node_id: str) -> list:
        """Retourne toutes les arêtes impliquant un nœud."""
        return [e for e in self._edges if e.source == node_id or e.target == node_id]

    def get_relations(self, node_id_a: str, node_id_b: str) -> list:
        """Retourne les relations entre deux nœuds."""
        return [
            e for e in self._edges
            if (e.source == node_id_a and e.target == node_id_b)
            or (e.source == node_id_b and e.target == node_id_a)
        ]

    @property
    def nodes(self) -> dict:
        return dict(self._nodes)

    @property
    def edges(self) -> list:
        return list(self._edges)
