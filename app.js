const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { predictImage } = require('./predict');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const PORT = 3000;

// Define the upload folder
const UPLOAD_FOLDER = 'static/uploads';
app.use('/static', express.static('static'));

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_FOLDER);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// Ensure the 'uploads' folder exists
fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });

// Define allowed extensions
const ALLOWED_EXTENSIONS = new Set(['jpeg', 'jpg', 'png']);

function allowedFile(filename) {
  const ext = path.extname(filename).toLowerCase().slice(1);
  return ALLOWED_EXTENSIONS.has(ext);
}

app.set('view engine', 'ejs');

app.get('/', (req, res) => {
  res.render('index');
});

app.post('/', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.redirect('/');
    }

    const filepath = path.join(__dirname, UPLOAD_FOLDER, req.file.filename);

    if (!allowedFile(req.file.filename)) {
      return res.redirect('/');
    }

    const result = await predictImage(filepath);
    const imageDimensions = await getImageDimensions(filepath);

    res.render('prediction', { result: result, imagePath: `/static/uploads/${req.file.filename}`, imageDimensions: imageDimensions });
  } catch (error) {
    console.error('Error in upload_file route:', error);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

async function getImageDimensions(filepath) {
  const img = await tf.node.decodeImage(fs.readFileSync(filepath));
  const { shape } = img;
  return { width: shape[1], height: shape[0] };
}
