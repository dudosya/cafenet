import torch
from torchvision import transforms

import dataset
import trainer
import model

# Define the necessary transformations including resizing to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet
])

folder_path = "./data"

real_dataset = dataset.RealCancerDataset(folder_path, transform=transform)


train_size = int(0.8 * len(real_dataset))
test_size = len(real_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(real_dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_classes = 4
MyModel = model.CaFeNet(num_classes).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MyModel.parameters(), lr=0.001)

MyTrainer = trainer.Trainer(MyModel,train_loader,test_loader,criterion,optimizer,device)
MyTrainer.train(num_epochs=2)