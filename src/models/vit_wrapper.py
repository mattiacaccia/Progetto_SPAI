
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = {
    "tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",     # 5.7M parametri, token 192-dim
    "small": "vit_small_patch16_224.augreg_in21k_ft_in1k",   # 22M parametri, token 384-dim
    "small32": "vit_small_patch32_224.augreg_in21k_ft_in1k", # patch 32: 4x meno token, risoluzione più grossolana
}

# parametri che definiscono l'architettura (salvati nel checkpoint per ricostruire il modello in test/tracking)
ARCH_KEYS = ("backbone", "img_size", "pool", "freeze_blocks", "num_queries", "auxiliary_loss")


class ViTFeatureBackbone(nn.Module):

    def __init__(self, timm_name: str, img_size: int = 640, pretrained: bool = True, pool: int = 2, freeze_blocks: int = 0):
        super().__init__()
        import timm

        # img_size=640: timm ricampiona il pos-embedding 14x14 (224px) alla griglia 40x40 quando carica i pesi
        self.model = timm.create_model(timm_name, pretrained=pretrained, img_size=img_size, num_classes=0, global_pool="")
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.pool = int(pool)
        self.img_size = img_size
        cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        mean = torch.tensor(cfg.get("mean", (0.5, 0.5, 0.5)), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("std", (0.5, 0.5, 0.5)), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.intermediate_channel_sizes = [int(self.model.num_features)]
        self.freeze(freeze_blocks)

    def freeze(self, n_blocks: int):
        
        if n_blocks <= 0:
            return
        for p in self.model.patch_embed.parameters():
            p.requires_grad_(False)
        if getattr(self.model, "cls_token", None) is not None:
            self.model.cls_token.requires_grad_(False)
        if getattr(self.model, "pos_embed", None) is not None:
            self.model.pos_embed.requires_grad_(False)
        for blk in list(self.model.blocks)[:n_blocks]:
            for p in blk.parameters():
                p.requires_grad_(False)

    def forward(self, pixel_values, pixel_mask):
        x = (pixel_values - self.mean) / self.std
        tokens = self.model.forward_features(x)               
        tokens = tokens[:, self.model.num_prefix_tokens:]       
        B, N, C = tokens.shape
        h, w = pixel_values.shape[-2] // self.patch, pixel_values.shape[-1] // self.patch
        fmap = tokens.transpose(1, 2).reshape(B, C, h, w)
        if self.pool > 1:
            fmap = F.avg_pool2d(fmap, self.pool)
        mask = F.interpolate(pixel_mask[None].float(), size=fmap.shape[-2:]).to(torch.bool)[0]
        return [(fmap, mask)]


class DETRViTWrapper(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "tiny", #di default tiny ma si può cambiare in small dal training_config.yaml
        img_size: int = 640,
        pool: int = 2,
        freeze_blocks: int = 4,
        num_queries: int = 100,
        auxiliary_loss: bool = True,
        init_transformer: str = "coco",
        load_pretrained: bool = True,
        model_name=None,  
    ):
        super().__init__()
        from transformers import DetrConfig, DetrForObjectDetection

        if backbone not in BACKBONES:
            raise ValueError(f"backbone '{backbone}' sconosciuto. Scegli tra {list(BACKBONES)}")
        if init_transformer not in ("coco", "random"):
            raise ValueError("init_transformer deve essere 'coco' o 'random'")

        self.arch = dict(backbone=backbone, img_size=img_size, pool=pool, freeze_blocks=freeze_blocks,
                         num_queries=num_queries, auxiliary_loss=auxiliary_loss)

        # DETR "vuoto": la ResNet creata di default da HF viene sostituita subito dal ViT
        config = DetrConfig(num_queries=num_queries, auxiliary_loss=auxiliary_loss)
        config.num_labels = num_classes
        detr = DetrForObjectDetection(config)

        vit = ViTFeatureBackbone(BACKBONES[backbone], img_size=img_size, pretrained=load_pretrained,
                                 pool=pool, freeze_blocks=freeze_blocks)
        detr.model.backbone = vit
        proj = nn.Conv2d(vit.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)
        nn.init.xavier_uniform_(proj.weight, gain=1.0)
        nn.init.zeros_(proj.bias)
        detr.model.input_projection = proj

        if load_pretrained and init_transformer == "coco":
            self._init_transformer_from_coco(detr)

        self.model = detr

    @staticmethod
    def _init_transformer_from_coco(detr):
        from transformers import DetrForObjectDetection
        src = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        prefixes = ("model.encoder.", "model.decoder.", "model.query_position_embeddings.", "bbox_predictor.")
        own = detr.state_dict()
        sd = {k: v for k, v in src.state_dict().items() if k.startswith(prefixes) and k in own and own[k].shape == v.shape}
        n_expected = sum(1 for k in own if k.startswith(prefixes))
        detr.load_state_dict(sd, strict=False)
        print(f"[DETR-ViT] Transformer inizializzato da COCO: {len(sd)}/{n_expected} tensori copiati")
        del src
        gc.collect()

    def forward(self, pixel_values, labels=None):
        if labels is not None:
            return self.model(pixel_values=pixel_values, labels=labels)
        return self.model(pixel_values=pixel_values)

    def param_groups(self, lr: float, backbone_lr: float, weight_decay: float):
        bb_ids = {id(p) for p in self.model.model.backbone.parameters()}
        bb = [p for p in self.parameters() if p.requires_grad and id(p) in bb_ids]
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in bb_ids]
        return [
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
            {"params": bb, "lr": backbone_lr, "weight_decay": weight_decay},
        ]

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


ViTDetWrapper = DETRViTWrapper