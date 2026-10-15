import argparse

from torchgeo.tasks import SemanticSegmentation
from torchgeo.models import ResNet18_Weights
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from perda import PerdaDataModule, Perda


ROOT = "./"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10)",
    )
    args = parser.parse_args()

    # Download, check and unpack the three splits into data/
    for split in ("train", "val", "test"):
        ds = Perda(root=f"{ROOT}/data", split=split, download=False)
        print(split, len(ds), "samples")

    weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

    dm = PerdaDataModule(
        root=f"{ROOT}/data",
        weights=weights,
        batch_size=8,
        num_workers=2,
    )

    task = SemanticSegmentation(
        model="unet",
        backbone="resnet18",
        weights=weights,
        in_channels=13,
        num_classes=len(Perda.classes),
        ignore_index=Perda.ignore_index,
        freeze_backbone=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ROOT}/models/semantic_segmentation",
        filename="Unet_ResNet18_SENTINEL2_ALL_MOCO_weights",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(task, datamodule=dm)
    trainer.test(task, datamodule=dm)


if __name__ == "__main__":
    main()