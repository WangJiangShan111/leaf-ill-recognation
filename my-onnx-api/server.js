const express = require('express');
const multer = require('multer');
const ort = require('onnxruntime-node'); 
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' }); 

let session;
ort.InferenceSession.create(path.join(__dirname, 'models', 'model.onnx'))
  .then((sess) => {
    session = sess;
    console.log('ONNX 模型加载成功');
  })
  .catch((err) => console.error('模型加载失败', err));

// 上传图片并进行推理
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!session) {
      return res.status(500).send('模型尚未加载');
    }

    const imagePath = req.file.path;
    const tensor = await preprocessImage(imagePath); 

    const feeds = { input: tensor }; 
    const results = await session.run(feeds);
    const output = results.output; 

    res.json({ predictions: output.data });

    fs.unlinkSync(imagePath);
  } catch (error) {
    console.error('推理错误', error);
    res.status(500).send('推理失败');
  }
});

async function preprocessImage(imagePath) {
  return new ort.Tensor('float32', new Float32Array([/*...*/]), [1, 3, 224, 224]); 
}

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`服务器正在运行，端口 ${port}`);
});
