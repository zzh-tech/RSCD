import os
import time
from skimage import io

from .metrics import *
from .image_proc import *

class Generic_train_test():
	def __init__(self, model, opts, dataloader, logger, dataloader_val=None):
		self.model=model
		self.opts=opts
		self.dataloader=dataloader
		self.logger=logger
		self.dataloader_val = dataloader_val

	def decode_input(self, data):
		raise NotImplementedError()

	def validation(self):
		raise NotImplementedError()

	def train_single_iterate(self, data, total_steps, epoch):
		_input=self.decode_input(data)

		self.model.set_input(_input)
		self.model.optimize_parameters()

		#=========== visualize results ============#
		if total_steps % self.opts.log_freq==0:
			info = self.model.get_current_scalars()
			for tag, value in info.items():
				self.logger.add_scalar(tag, value, total_steps)

			results = self.model.get_current_visuals()
			for tag, images in results.items():
				self.logger.add_images(tag, images, total_steps)

			print('epoch', epoch, 'steps', total_steps)
			print('losses', info)

	def train(self):
		total_steps = 0
		if self.dataloader is not None:
			print('#training images ', len(self.dataloader)*self.opts.batch_sz)

		for epoch in range(self.opts.start_epoch, self.opts.max_epochs):
			if epoch > self.opts.lr_start_epoch_decay - self.opts.lr_step:
				self.model.update_lr()

			if epoch % self.opts.save_freq==0:
				self.model.save_checkpoint(str(epoch))

			if self.dataloader is not None:
				for i, data in enumerate(self.dataloader):
					total_steps+=1
					self.train_single_iterate(data, total_steps, epoch)
			else:
				for i in range(10000):
					total_steps+=1
					self.train_single_iterate(None, total_steps, epoch)

			# validation if dataloader provided
			if self.dataloader_val is not None:
				self.validation(epoch)

	def train_single_instance(self):
		total_steps = 0
		data=iter(self.dataloader).next()

		for epoch in range(10000):
			for i in range(1000):
				total_steps+=1
				self.train_single_iterate(data, total_steps, epoch)
				


