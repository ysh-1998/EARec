import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CFTower(nn.Module):
    def __init__(self, cf_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.cf_tower_name = cf_tower
        # self.select_layer = args.mm_vision_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_cf_tower', False):
            self.load_model()
        # else:
        #     self.cfg_only = CLIPVisionConfig.from_pretrained(self.cf_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.cf_tower_name))
            return

        # self.image_processor = CLIPImageProcessor.from_pretrained(self.cf_tower_name)
        # self.cf_tower = CLIPVisionModel.from_pretrained(self.cf_tower_name, device_map=device_map)
        self.cf_tower = nn.Embedding.from_pretrained(torch.load(self.cf_tower_name))
        self.cf_tower.requires_grad_(False)

        self.is_loaded = True

    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    @torch.no_grad()
    def forward(self, items):
        if type(items) is list:
            item_features = []
            for item in items:
                item_feature = self.cf_tower(item.to(device=self.device, dtype=self.dtype).unsqueeze(0)) # , output_hidden_states=True)
                # item_feature = self.feature_select(item_forward_out).to(item.dtype)
                item_features.append(item_feature)
        else:
            item_features = self.cf_tower(items) # , output_hidden_states=True)
            # item_features = self.feature_select(item_forward_outs).to(items.dtype)

        return item_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.cf_tower.dtype

    @property
    def device(self):
        return self.cf_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.cf_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # return self.config.hidden_size
        return self.cf_tower.embedding_dim

    # @property
    # def num_patches_per_side(self):
    #     return self.config.image_size // self.config.patch_size

    # @property
    # def num_patches(self):
    #     return (self.config.image_size // self.config.patch_size) ** 2



class CFTowerS2(CFTower):
    def __init__(self, cf_tower, args, delay_load=False):
        super().__init__(cf_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_cf_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.cf_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.cf_tower_name)
        self.cf_tower = CLIPVisionModel.from_pretrained(self.cf_tower_name, device_map=device_map)
        self.cf_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.cf_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
