"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import torch
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist,preprocess
from art.metrics import clever_u,clever_t,clever
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from art.utils import load_cifar10
from torchvision.models.mobilenet import mobilenet_v2
from art.visualization import create_sprite, convert_to_rgb, save_image, plot_3d
import time
# Step 0: Define the neural network model, return logits instead of activation in forward method
R_L1 = 40
R_L2 = 2
R_LI = 0.1

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		self.use_res_connect = self.stride == 1 and inp == oup

		self.conv = nn.Sequential(
			# pw
			nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
			nn.BatchNorm2d(inp * expand_ratio),
			nn.ReLU6(inplace=True),
			# dw
			nn.Conv2d(
				inp * expand_ratio,
				inp * expand_ratio,
				3,
				stride,
				#1,
				groups=inp * expand_ratio,
				bias=False),
			nn.BatchNorm2d(inp * expand_ratio),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
			nn.BatchNorm2d(oup),
		)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		return self.conv(x)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv_1 = InvertedResidual(3, 4, stride=1, expand_ratio=5)
		# self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
		self.conv_2 = InvertedResidual(4, 10, stride=1, expand_ratio=5)
		# self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
		# self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
		self.fc_1 = nn.Linear(in_features=6 * 6 * 10, out_features=100)
		self.fc_2 = nn.Linear(in_features=100, out_features=10)

	def forward(self, x):
		x = F.relu(self.conv_1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv_2(x))
		x = F.max_pool2d(x, 2, 2)
		# x = x.view(-1, 4 * 4 * 10)
		x = x.view(-1, 6 * 6 * 10)
		x = F.relu(self.fc_1(x))
		x = self.fc_2(x)
		return x


# Step 1: Load the CIFAR10 dataset
class CIFAR10_dataset(Dataset):
	def __init__(self, data, targets, transform=None):
		self.data = data
		self.targets = torch.LongTensor(targets)
		self.transform = transform

	def __getitem__(self, index):
		x = Image.fromarray(((self.data[index] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
		x = self.transform(x)
		y = self.targets[index]
		return x, y

	def __len__(self):
		return len(self.data)


# Step 1: Load the CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

cifar_mu = np.ones((3, 32, 32))
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465

# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

cifar_std = np.ones((3, 32, 32))
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616

x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

transform = transforms.Compose(
	[transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
)

dataset = CIFAR10_dataset(x_train, y_train, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Step 2: Create the model

model = Net()

# model = mobilenet_v2(num_classes = 10)

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
	model=model,
	clip_values=(0.0, 1.0),
	preprocessing=(cifar_mu, cifar_std),
	loss=criterion,
	optimizer=optimizer,
	input_shape=(3, 32, 32),
	nb_classes=10,

)

# Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=10)
exp_time = time.strftime('%H_%M_%S')
# torch.save(classifier.model.state_dict(), 'pth/{}.pth.tar'.format(exp_time))
# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)

attack_pgd = ProjectedGradientDescent(
	classifier,
	norm=np.inf,
	eps=8.0 / 255.0,
	eps_step=2.0 / 255.0,
	max_iter=40,
	targeted=False,
	num_random_init=5,
	batch_size=32,
)

x_test_adv = attack.generate(x=x_test)
# x_test_adv = attack_pgd.generate(x_test)
# np.save('./adv.npy', x_test_adv)
# x_test_adv = np.load('./adv.npy')

# Step 7: Evaluate the ART classifier on adversarial test        examples
# x_save = x_test[0:100]
# x_adv_save = x_test_adv[0:100]
# x_sprite = create_sprite(x_save)
# x_adv_sprite = create_sprite(x_adv_save)
# f_name = "./test_sprite_cifar.png"
# f_adv_name = "./test_adv_sprite_cifar.png"
# save_image(x_sprite, f_name)
# save_image(x_adv_sprite, f_adv_name)


predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

# Test targeted clever
# res0 = clever_t(classifier, x_test[-1], 2, 10, 5, R_L1, norm=1, pool_factor=3)
# res1 = clever_t(classifier, x_test[-1], 2, 10, 5, R_L2, norm=2, pool_factor=3)
# res2 = clever_t(classifier, x_test[-1], 2, 10, 5, R_LI, norm=np.inf, pool_factor=3)
# print("Targeted PyTorch: %f %f %f", res0, res1, res2)

# Test untargeted clever
# res0 = clever_u(classifier, x_test[-1], 10, 5, R_L1, norm=1, pool_factor=3)
# res1 = clever_u(classifier, x_test[-1], 10, 5, R_L2, norm=2, pool_factor=3)
total = 0
for i in range(10):
	res = clever_u(classifier, x_test[-i], 50, 10, R_LI, norm=np.inf, pool_factor=3)
	total+=res
	print("Untargeted PyTorch: i: %d, res:%f",i, res)
print('==============================\n', total/10)
total = 0
for i in range(10):
	res = clever_u(classifier, x_test[i], 50, 10, R_LI, norm=np.inf, pool_factor=3)
	total+=res
	print("Untargeted PyTorch: i: %d, res:%f",i, res)
print('=============================\n', total/10)