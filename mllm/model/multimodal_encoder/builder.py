import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .cf_encoder import CFTower, CFTowerS2


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_cf_tower(cf_tower_cfg, **kwargs):
    cf_tower = getattr(cf_tower_cfg, 'mm_cf_tower', getattr(cf_tower_cfg, 'cf_tower', None))
    is_absolute_path_exists = os.path.exists(cf_tower)
    use_s2 = getattr(cf_tower_cfg, 's2', False)
    if is_absolute_path_exists or cf_tower.startswith("openai") or cf_tower.startswith("laion") or "ShareGPT4V" in cf_tower:
        if use_s2:
            return CFTowerS2(cf_tower, args=cf_tower_cfg, **kwargs)
        else:
            return CFTower(cf_tower, args=cf_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {cf_tower}')