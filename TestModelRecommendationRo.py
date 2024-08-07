import os
import lightning.pytorch as pl

# os.environ['MODLEE_API_KEY'] = "0licKyh5qcDgZlvlGbM1uwfly0qihoei" #School Rohit
# os.environ['MODLEE_API_KEY'] = "E1S58A6F4dUUBJEG02E1R1TG631i8b8E" #Personal Rohit
os.environ['MODLEE_API_KEY'] = "zBwhs4rPwopuja8zMi0taaqHE3I7qbyK" #Pranjal

import torch, torchvision
import torchvision.transforms as transforms

import modlee
if os.environ.get('MODLEE_API_KEY') is None:
    print("Module key not set")
else:
    modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))
    print("Module initialized")

transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms)
val_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transforms)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
   )
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16
)

recommender = modlee.recommender.from_modality_task(
    modality='image',
    task='classification',
    )
# breakpoint()
recommender.fit(train_dataloader)
modlee_model = recommender.model
print(f"\nRecommended model: \n{modlee_model}")

with modlee.start_run() as run:
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(
        model=modlee_model,
        train_dataloaders=train_dataloader
        )

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")
artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = sorted(os.listdir(artifacts_path))
print(f"Saved artifacts: {artifacts}")