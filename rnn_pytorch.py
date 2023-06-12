import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 28  # 1x28 1 dimension as a sequence and other as input(feature size)
sequence_len = 28
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
	plt.subplot(2, 3, i + 1)
	plt.imshow(example_data[i][0], cmap='gray')
plt.show()


class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(RNN, self).__init__()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		# self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
		# self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		# x -> batch_size, seq, input_size
		self.fc = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		# out, _ = self.rnn(x, h0)
		# out, _ = self.gru(x, h0)
		out, _ = self.lstm(x, (h0, c0))
		# out: batch_size, seq_length, hidden_size
		# out (N, 28, 128)
		out = out[:, -1, :]
		# out (N, 128)
		out = self.fc(out)
		return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# origin shape: [100, 1, 28, 28]
		# resized: [100, 28, 28]
		images = images.reshape(-1, sequence_len, input_size).to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 100 == 0:
			print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for images, labels in test_loader:
		images = images.reshape(-1, sequence_len, input_size).to(device)
		labels = labels.to(device)
		outputs = model(images)
		# max returns (value ,index)
		_, predicted = torch.max(outputs.data, 1)
		n_samples += labels.size(0)
		n_correct += (predicted == labels).sum().item()

	acc = 100.0 * n_correct / n_samples
	print(f'Accuracy of the network on the 10000 test images: {acc} %')