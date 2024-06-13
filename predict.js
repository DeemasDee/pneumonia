const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function predictImage(filepath) {
  try {
    // Load the model
    const model = await tf.loadGraphModel('file://model_tfjs/model.json'); // Replace with the correct path

    // Preprocess the image
    const imageBuffer = fs.readFileSync(filepath);
    let img = tf.node.decodeImage(imageBuffer, 3); // Ensure 3 channels
    img = tf.image.resizeBilinear(img, [150, 150]);
    img = img.expandDims(0);
    img = img.div(tf.scalar(255.0));

    // Predict the image
    const prediction = model.predict(img);
    const result = (prediction.dataSync()[0] >= 0.5) ? "PNEUMONIA" : "NORMAL";

    return result;
  } catch (error) {
    console.error('Error in predictImage:', error);
    throw error;
  }
}

module.exports = { predictImage };
