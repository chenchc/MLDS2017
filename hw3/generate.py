from src.model_began import ModelBEGAN

if __name__ == '__main__':
	model = ModelBEGAN(test_mode=True)
	model.test('data/model_began/model.ckpt-136')
