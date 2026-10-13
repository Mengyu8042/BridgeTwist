"""CLIP text likelihood for CelebA-HQ."""

import torch
import torch.nn.functional as F

from utilities import LikelihoodFunction


class CLIPTextLikelihood(LikelihoodFunction):
    """
    CLIP-based text likelihood for text-conditional CelebA-HQ generation.

    Uses OpenCLIP (convnext_xxlarge) to measure image-text alignment:

        log p(text | x_1) = log_softmax(
            [T · max_j cos(img_feat, pos_feat_j),
             T · max_k cos(img_feat, neg_feat_k)]
        )[0]

    where T is the fixed likelihood temperature (default 100).

    The CLIP model and precomputed text features are cached after the first call,
    so loading cost is paid only once per CLIPTextLikelihood instance.

    Gradient flow through encode_image is preserved (model params frozen,
    but input gradients still flow), making this usable as an SMC twist function.
    """

    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        positive_texts: list,
        negative_texts: list,
        image_shape: tuple,
        clip_model_path: str,
        device: str,
        temperature: float = 100.0,
        clip_input_size: int = 256,
    ):
        """
        Args:
            positive_texts:   e.g. ["a photo of a woman"]
            negative_texts:   e.g. ["a photo of a man", ...]
            image_shape:      (C, H, W) of the flat particles
            clip_model_path:  path to open_clip_pytorch_model.bin
            device:           torch device string
            temperature:      logit scale T (default 100)
            clip_input_size:  CLIP input resolution (256 for convnext_xxlarge)
        """
        self.positive_texts   = positive_texts
        self.negative_texts   = negative_texts
        self.image_shape      = image_shape
        self.clip_model_path  = clip_model_path
        self.device           = device
        self.temperature      = temperature
        self.clip_input_size  = clip_input_size

        # Populated on first call to _load_clip()
        self._clip_model    = None
        self._pos_features  = None   # (n_pos, d)
        self._neg_features  = None   # (n_neg, d)

    # ------------------------------------------------------------------
    def _load_clip(self):
        """Lazy-load the CLIP model and precompute frozen text features."""
        if self._clip_model is not None:
            return
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required for CLIPTextLikelihood. "
                "Create and activate the shared environment specified in environment.yml."
            )
        model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=self.clip_model_path
        )
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        tokenizer = open_clip.get_tokenizer("convnext_xxlarge")
        pos_tok = tokenizer(self.positive_texts).to(self.device)
        neg_tok = tokenizer(self.negative_texts).to(self.device)

        with torch.no_grad():
            pos_f = model.encode_text(pos_tok)
            self._pos_features = (
                pos_f / pos_f.norm(dim=-1, keepdim=True)
            ).detach()                                          # (n_pos, d)

            neg_f = model.encode_text(neg_tok)
            self._neg_features = (
                neg_f / neg_f.norm(dim=-1, keepdim=True)
            ).detach()                                          # (n_neg, d)

        self._clip_model = model
        print(
            f"  CLIPTextLikelihood: loaded convnext_xxlarge  "
            f"n_pos={len(self.positive_texts)}  "
            f"n_neg={len(self.negative_texts)}"
        )

    # ------------------------------------------------------------------
    def _preprocess(self, x_img: torch.Tensor) -> torch.Tensor:
        """Convert (K, C, H, W) in [-1, 1] → CLIP-normalised tensor."""
        x = (x_img + 1.0) / 2.0                                # [0, 1]
        mean = x_img.new_tensor(self._CLIP_MEAN).view(1, 3, 1, 1)
        std  = x_img.new_tensor(self._CLIP_STD ).view(1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[-2] != self.clip_input_size or x.shape[-1] != self.clip_input_size:
            x = F.interpolate(
                x, size=(self.clip_input_size, self.clip_input_size),
                mode="bilinear", align_corners=False,
            )
        return x

    # ------------------------------------------------------------------
    def log_prob(self, y_unused: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(text | x_1) for each particle.

        Args:
            y_unused: ignored (text stored in self.positive_texts)
            x_flat:   (K, d) flat image particles

        Returns:
            log_probs: (K,) — log-probability of the positive description
        """
        self._load_clip()
        K = x_flat.shape[0]
        C, H, W = self.image_shape
        x_img  = x_flat.reshape(K, C, H, W)
        x_prep = self._preprocess(x_img)                        # (K, C, 256, 256)

        # encode_image stays in the autograd graph →
        # gradients flow back to x_flat even though model params are frozen.
        img_feat = self._clip_model.encode_image(x_prep)        # (K, d)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        pos_sim = (img_feat @ self._pos_features.T).max(dim=1)[0]  # (K,)
        neg_sim = (img_feat @ self._neg_features.T).max(dim=1)[0]  # (K,)

        logits    = torch.stack(
            [self.temperature * pos_sim, self.temperature * neg_sim], dim=1
        )                                                        # (K, 2)
        log_probs = F.log_softmax(logits, dim=1)[:, 0]          # (K,)
        return log_probs

    def set_prompts(self, positive_texts: list, negative_texts: list):
        """Reuse the frozen CLIP model while changing the prompt's text features."""
        if not positive_texts or not negative_texts:
            raise ValueError("Both positive and negative descriptions are required")
        self.positive_texts = positive_texts
        self.negative_texts = negative_texts
        if self._clip_model is None:
            self._load_clip()
            return
        import open_clip

        tokenizer = open_clip.get_tokenizer("convnext_xxlarge")
        with torch.no_grad():
            pos_f = self._clip_model.encode_text(tokenizer(positive_texts).to(self.device))
            neg_f = self._clip_model.encode_text(tokenizer(negative_texts).to(self.device))
            self._pos_features = (pos_f / pos_f.norm(dim=-1, keepdim=True)).detach()
            self._neg_features = (neg_f / neg_f.norm(dim=-1, keepdim=True)).detach()

    @torch.no_grad()
    def clip_l_values(self, samples: torch.Tensor) -> torch.Tensor:
        """Positive-text cosine similarity for each final resampled image.

        Evaluate one image at a time.
        Use the floating-point endpoint samples before PNG clipping/quantization.
        """
        self._load_clip()
        values = []
        for img in samples:
            x = self._preprocess(img.unsqueeze(0).to(self.device))
            feat = self._clip_model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            values.append((feat @ self._pos_features.T).max(dim=1)[0])
        return torch.cat(values)
