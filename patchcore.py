from anomalib.data import Folder
from anomalib.engine import Engine
from MH_PatchCore_lightning_model import Patchcore


# Initialize model and data

folder_datamodule = Folder(
    name ="bottle", 
    root="C:/Users/gowth/Desktop/Projet perso/PatchCore/datasets/bottle/",
    normal_dir="train/good_sub",
    abnormal_dir="test/broken_small",
    normal_split_ratio=0.2,
    test_split_ratio=0.2,
    val_split_ratio=0.2,
    extensions= [".png"],
    num_workers=0
)
folder_datamodule.setup()
train_dataloader = folder_datamodule.train_dataloader()
val_dataloader = folder_datamodule.val_dataloader()
test_dataloader = folder_datamodule.test_dataloader()

print(len(train_dataloader))
print(len(val_dataloader))
print(len(test_dataloader))

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1
)

# Train using the Engine
engine = Engine()
engine.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
