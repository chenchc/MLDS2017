from models.bilstm_hybrid import bilstm_hybrid

model = bilstm_hybrid()
model.test('data/model_bilstm_hybrid6/model.ckpt')

