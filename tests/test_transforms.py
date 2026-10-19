# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
import pytest
import torch
from tests.conftest import create_images
from torch.utils.data import DataLoader

from autovision.augment import (
    GaussianBlurArgs,
    MethodTransformArgs,
    MultiViewTransform,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    ViewTransform,
)
from autovision.augment.transform_args import (
    IMAGENET_NORMALIZE_MEAN,
    IMAGENET_NORMALIZE_STD,
)


@pytest.fixture
def create_images_fixture(tmp_path):
    """Create 8 small images and matching resolved transform args."""
    paths = create_images(tmp_path / "images", num_images=8, size=(48, 48))
    args = MethodTransformArgs(
        image_size=(32, 32),
        random_resize=RandomResizeArgs(min_scale=0.5, max_scale=1.0),
        random_flip=RandomFlipArgs(),
        normalize=NormalizeArgs(),
    )
    args.resolve_auto_fields()
    return paths, args


def make_args(**overrides) -> MethodTransformArgs:
    defaults = dict(
        image_size=(32, 32),
        random_resize=RandomResizeArgs(min_scale=0.5, max_scale=1.0),
        random_flip=RandomFlipArgs(),
        color_jitter=None,
        random_gray_scale=None,
        gaussian_blur=None,
        solarize=None,
        normalize=NormalizeArgs(),
    )
    defaults.update(overrides)
    return MethodTransformArgs(**defaults)


class TestTransformArgs:
    def test_resolve_auto_fields_sets_num_channels(self):
        args = make_args()
        assert args.num_channels == "auto"
        args.resolve_auto_fields()
        assert args.num_channels == 3

    def test_imagenet_defaults(self):
        assert IMAGENET_NORMALIZE_MEAN == (0.485, 0.456, 0.406)
        assert IMAGENET_NORMALIZE_STD == (0.229, 0.224, 0.225)

    def test_resolve_incompatible_channels_repeats_normalize(self):
        args = make_args(normalize=NormalizeArgs(mean=(0.5,), std=(0.25,)))
        args.resolve_auto_fields()
        assert args.num_channels == 1

    def test_resolve_incompatible_disables_photometric_for_non_rgb(self):
        args = make_args(
            color_jitter=None,
            normalize=NormalizeArgs(mean=(0.5,), std=(0.25,)),
            gaussian_blur=GaussianBlurArgs(),
            solarize=None,
        )
        args.resolve_auto_fields()
        args.resolve_incompatible()
        assert args.num_channels == 1
        assert args.color_jitter is None
        assert args.random_gray_scale is None
        assert args.solarize is None
        # blur is channel-safe and stays
        assert args.gaussian_blur is not None

    def test_normalize_roundtrip(self):
        args = NormalizeArgs()
        restored = NormalizeArgs.from_dict(args.to_dict())
        assert restored == args


class TestViewTransform:
    def test_output_shape_and_range(self):
        transform = ViewTransform(make_args())
        image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        out = transform(image)
        assert out.shape == (3, 32, 32)
        assert out.dtype == torch.float32
        # Normalized output is roughly centered; just check finiteness.
        assert torch.isfinite(out).all()

    def test_deterministic_under_seed(self):
        import random as _random

        transform = ViewTransform(
            make_args(color_jitter=None, gaussian_blur=GaussianBlurArgs(prob=1.0))
        )
        image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        torch.manual_seed(0)
        _random.seed(0)
        out1 = transform(image)
        torch.manual_seed(0)
        _random.seed(0)
        out2 = transform(image)
        assert torch.allclose(out1, out2)

    def test_flip_probs_zero(self):
        transform = ViewTransform(
            make_args(random_flip=RandomFlipArgs(horizontal_prob=0.0, vertical_prob=0.0))
        )
        image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        _ = transform(image)

    def test_multi_view(self):
        args = make_args()
        multi = MultiViewTransform(
            [ViewTransform(args), ViewTransform(args)], transform_args=args
        )
        out = multi({"image": torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)})
        assert len(out) == 2
        assert all(view["image"].shape == (3, 32, 32) for view in out)


class TestImageDataset:
    def test_dataset_and_collate(self, create_images_fixture):
        paths, args = create_images_fixture
        from autovision.data import ImageDataset

        multi = MultiViewTransform(
            [ViewTransform(args), ViewTransform(args)], transform_args=args
        )
        dataset = ImageDataset(
            image_dir=None,
            image_filenames=[str(p) for p in paths],
            transform=multi,
        )
        assert len(dataset) == len(paths)
        item = dataset[0]
        assert set(item.keys()) == {"filename", "views"}
        assert len(item["views"]) == 2
        assert item["views"][0].shape == (3, 32, 32)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert len(batch["views"]) == 2
        assert batch["views"][0].shape == (4, 3, 32, 32)
        assert len(batch["filename"]) == 4
