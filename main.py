import torch
import cv2
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import Network
from glob import glob
import fnmatch

# load dataset from https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
# set up device

writer = SummaryWriter()
device = "cuda" if torch.cuda.is_available() else "cpu"

# data preprocessing
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50, 50)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalize
])

# data loading
image_patches = glob("./archive/**/*.png", recursive=True)

pattern_zero = "*class0.png"
patter_one = "*class1.png"

class_zero = fnmatch.filter(image_patches, pattern_zero)
class_one = fnmatch.filter(image_patches, patter_one)
target = []

for target_value in class_zero:
    target.append(0)
for target_value in class_one:
    target.append(1)

dataset = [[*class_zero, *class_one], target]


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.data[0][index]
        target = self.data[1][index]
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data[0])


dataset = MyDataset(dataset, transform)
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])


batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# model evaluation
model = Network(500, 230, 80).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # if necessery add weight_decay=0.001
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 20


def train(epoch):
    model.train()
    running_loss = 0.0
    running_correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data = Variable(data.to(device))
        target = Variable(target.to(device))
        out = model(data)
        optimizer.zero_grad()
        criterion = F.cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(out.data, dim=1)
        running_correct += (predicted == target).sum().item()
        if (batch_id + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_id + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + batch_id)
            running_accuracy = running_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', running_accuracy, epoch * len(train_loader) + batch_id)
            running_correct = 0
            running_loss = 0.0


if __name__ == '__main__':
    for epoch in range(num_epochs):
        train(epoch)
        scheduler.step()

    with torch.no_grad():
        model.eval()
        correct = 0
        loss = 0
        for data, target in test_loader:
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum").item()
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == target).sum().item()
        print("mean loss: ", loss / len(test_loader))
        print("correct in %: ", 100. * correct / len(test_loader.dataset), "%")

    writer.close()
