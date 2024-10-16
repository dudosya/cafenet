import torch
from tqdm import tqdm  # Import tqdm for progress bar

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            criterion (nn.Module): Loss function.
            optimizer (Optimizer): Optimizer for training.
            device (torch.device): Device to run the training on.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Wrap the train loader with tqdm to display progress
        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device).float()
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()
            outputs, features = self.model(images, labels)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Wrap the test loader with tqdm to display progress
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device).float()
                labels = labels.to(self.device).long()

                outputs, features = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
