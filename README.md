# IMAGINE — Photo Editor Interactif Objet par Objet

Interactive photo editor with per-object manipulation, powered by a hybrid LLM
(local Qwen 14B + cloud Gemini/Groq) and a continuous Spatial Field for
light/material/depth physics.

## Architecture

```
photo_editor/
├── pipeline.py              ← Orchestrator (PhotoEditorPipeline)
├── spatial_field.py         ← SpatialField — continuous light/material/depth
├── visual_memory.py         ← VisualMemory — 8D signatures + scene_hash
├── scene_graph.py           ← SceneGraph light — topology only
├── semantic_encoder.py      ← SemanticEncoder — metrics → text for LLM
├── hybrid_router.py         ← HybridRouter — Qwen/Gemini routing
├── modification_engine.py   ← ObjectModificationEngine
├── cavity_inpainter.py      ← CavityInpainter — CPU patch matching
└── tests/
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY / GROQ_API_KEY
```

```python
from photo_editor.pipeline import PhotoEditorPipeline
import numpy as np

image     = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
depth_map = np.random.rand(480, 640).astype(np.float32)

pipeline = PhotoEditorPipeline(image, depth_map)
pipeline.add_object("cup_01", mask, bbox, crop)

result = pipeline.process_modification(
    image=image, object_id="cup_01", mask=mask, crop=crop,
    delta_params={"color_delta": 0.1}, mask_moved=False,
)
```

## Running Tests

```bash
python -m pytest photo_editor/tests/ -v
```