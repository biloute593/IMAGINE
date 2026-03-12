"""
spatial_field.py — SpatialField continu
Représentation continue de la scène : lumière, matière, profondeur.
Remplace les effets physiques du SceneGraph (qui garde uniquement la topologie).

AGENT : sigma=15.0 est le paramètre clé de propagate_modification().
        Augmenter pour objets grands, réduire pour corrections locales.
        Ne pas dépasser 50.
"""

import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter


# Mapping matériau → (reflectivity, diffusion)
MATERIAL_DEFAULTS = {
    0: (0.05, 0.8),   # surface_generique
    1: (0.85, 0.1),   # metal_speculaire
    2: (0.70, 0.3),   # verre_texture
    3: (0.10, 0.9),   # fourrure_textile
    4: (0.05, 0.7),   # plastique_mat
}

MATERIAL_NAMES = {
    "surface_generique": 0,
    "metal_speculaire": 1,
    "verre_texture": 2,
    "fourrure_textile": 3,
    "plastique_mat": 4,
}


class SpatialField:
    """
    Champ continu représentant les propriétés physiques de chaque pixel.

    Attributs :
        depth_field        (H, W) float32  — profondeur depuis MiDaS
        light_field        (H, W) float32  — luminance locale [0,1]
        material_field     (H, W) int8     — index matériau
        reflectivity_field (H, W) float32  — 0=mat, 1=miroir
        diffusion_field    (H, W) float32  — dispersion lumière locale
    """

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.depth_field = np.zeros((height, width), dtype=np.float32)
        self.light_field = np.zeros((height, width), dtype=np.float32)
        self.material_field = np.zeros((height, width), dtype=np.int8)
        self.reflectivity_field = np.zeros((height, width), dtype=np.float32)
        self.diffusion_field = np.full((height, width), 0.8, dtype=np.float32)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def build_from_scene(
        self,
        depth_map: np.ndarray,
        segments: list,
        signatures: list,
    ) -> None:
        """
        Initialise tous les champs depuis MiDaS + SAM + signatures 8D.

        Args:
            depth_map  : tableau float32 (H, W) issu de MiDaS small.
            segments   : liste de masques binaires np.ndarray (H, W) bool.
            signatures : liste de dicts de signatures 8D (une par segment).
        """
        if depth_map.shape != (self.height, self.width):
            raise ValueError(
                f"depth_map shape {depth_map.shape} != "
                f"({self.height}, {self.width})"
            )

        # Normalise la depth map en [0, 1]
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            self.depth_field = (depth_map - d_min) / (d_max - d_min)
        else:
            self.depth_field = depth_map.copy()
        self.depth_field = self.depth_field.astype(np.float32)

        # Initialise light_field depuis depth (approximation : objets proches
        # reçoivent plus de lumière)
        self.light_field = (1.0 - self.depth_field).astype(np.float32)

        # Applique les propriétés par objet depuis les signatures
        for mask, sig in zip(segments, signatures):
            if mask is None or sig is None:
                continue
            material_name = sig.get("material_class", "surface_generique")
            material_id = MATERIAL_NAMES.get(material_name, 0)
            refl, diff = MATERIAL_DEFAULTS[material_id]
            brightness = float(sig.get("brightness", 0.5))

            self.material_field[mask] = material_id
            self.reflectivity_field[mask] = refl
            self.diffusion_field[mask] = diff
            self.light_field[mask] = np.clip(brightness, 0.0, 1.0)

        # Lissage pour éviter les transitions brusques aux bords de masque
        self.light_field = gaussian_filter(self.light_field, sigma=1.0)

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def sample(self, x: int, y: int) -> dict:
        """
        Retourne les propriétés physiques du pixel (x, y).

        Args:
            x : colonne (axe horizontal).
            y : ligne   (axe vertical).

        Returns:
            dict avec depth, luminance, material_id, reflectivity, light_diffusion.
        """
        y = int(np.clip(y, 0, self.height - 1))
        x = int(np.clip(x, 0, self.width - 1))
        return {
            "depth": float(self.depth_field[y, x]),
            "luminance": float(self.light_field[y, x]),
            "material_id": int(self.material_field[y, x]),
            "reflectivity": float(self.reflectivity_field[y, x]),
            "light_diffusion": float(self.diffusion_field[y, x]),
        }

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate_modification(
        self,
        object_mask: np.ndarray,
        delta_params: dict,
        sigma: float = 15.0,
    ) -> None:
        """
        Propage l'impact d'une modification (lumière/reflet/ombre) avec
        décroissance gaussienne locale.

        AGENT : sigma=15.0 par défaut. Augmenter pour grands objets.
                Ne pas dépasser 50.

        Args:
            object_mask  : masque binaire (H, W) de l'objet modifié.
            delta_params : corrections appliquées (dict du LLM).
            sigma        : rayon d'influence en pixels [1, 50].
        """
        # AGENT : protection sigma max
        sigma = float(np.clip(sigma, 1.0, 50.0))

        # --- Impact lumière ---
        # color_delta et lumiere_ambiante contribuent tous deux à la luminance ;
        # on les moyenne (× 0.5) pour éviter les surexpositions cumulées.
        color_delta = float(delta_params.get("color_delta", 0.0))
        lumiere = float(delta_params.get("lumiere_ambiante", 0.0))
        light_impact = (color_delta + lumiere) * 0.5

        if abs(light_impact) > 1e-6:
            # Crée un champ d'impact centré sur le masque de l'objet
            impact_map = np.where(object_mask, light_impact, 0.0).astype(np.float32)
            diffused = gaussian_filter(impact_map, sigma=sigma)
            self.light_field = np.clip(self.light_field + diffused, 0.0, 1.0)

        # --- Impact réflectivité ---
        refl_delta = float(delta_params.get("reflectivity_delta", 0.0))
        if abs(refl_delta) > 1e-6:
            impact_map = np.where(object_mask, refl_delta, 0.0).astype(np.float32)
            diffused = gaussian_filter(impact_map, sigma=sigma * 0.5)
            self.reflectivity_field = np.clip(
                self.reflectivity_field + diffused, 0.0, 1.0
            )

        # --- Impact ombre (shadow_intensity) ---
        shadow = float(delta_params.get("shadow_intensity", 0.0))
        if abs(shadow) > 1e-6:
            impact_map = np.where(object_mask, -shadow * 0.3, 0.0).astype(
                np.float32
            )
            diffused = gaussian_filter(impact_map, sigma=sigma * 1.5)
            self.light_field = np.clip(self.light_field + diffused, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Hachage d'état
    # ------------------------------------------------------------------

    def compute_scene_hash(self) -> str:
        """
        Hash MD5 du light_field + depth_field (état physique de la scène).
        Utilisé par VisualMemory pour invalider les corrections obsolètes.

        Returns:
            Chaîne hexadécimale MD5 de 32 caractères.
        """
        hasher = hashlib.md5()
        hasher.update(self.light_field.tobytes())
        hasher.update(self.depth_field.tobytes())
        return hasher.hexdigest()
