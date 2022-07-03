import React from 'react';
import * as tf from '@tensorflow/tfjs';

import { INCEPTION_CLASSES } from './labels';

export default function Mobilenet() {
  const imgRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const cropRef = React.useRef(null);
  const [predictions, setPredictions] = React.useState([]);

  React.useEffect(() => {
    const modelPath =
      'https://tfhub.dev/google/tfjs-model/imagenet/inception_v3/classification/3/default/1';
    tf.tidy(() => {
      tf.loadGraphModel(modelPath, { fromTFHub: true }).then((model) => {
        const tensor = tf.browser.fromPixels(imgRef.current);
        const redified = tf.image
          .resizeBilinear(tensor, [299, 299], true)
          .div(255)
          .reshape([1, 299, 299, 3]);
        const result = model.predict(redified);
        setPredictions(tf.topk(result, 3).indices.dataSync());
      });
    });

    tf.ready().then(() => {
      const modelPath = '/model/tfjs_quant_uint8/model.json';
      tf.tidy(() => {
        tf.loadLayersModel(modelPath).then((model) => {
          const tensor = tf.browser.fromPixels(imgRef.current);
          const redified = tf.image
            .resizeBilinear(tensor, [256, 256], true)
            .div(255)
            .reshape([1, 256, 256, 3]);
          const result = model.predict(redified);
          result.print();

          const imgWidth = imgRef.current.width;
          const imgHeight = imgRef.current.height;
          const box = result.dataSync();
          const startX = box[0] * imgWidth;
          const startY = box[1] * imgHeight;
          const width = (box[2] - box[0]) * imgWidth;
          const height = (box[3] - box[1]) * imgHeight;

          tf.browser.toPixels(tensor, canvasRef.current).then(() => {
            const ctx = canvasRef.current.getContext('2d');
            ctx.strokeStyle = '#0F0';
            ctx.lineWidth = 4;
            console.log(startX, startY, width, height);
            ctx.strokeRect(startX, startY, width, height);
          });

          const tHeight = tensor.shape[0];
          const tWidth = tensor.shape[1];
          const tStartX = box[0] * tWidth;
          const tStartY = box[1] * tHeight;
          const cropLength = Math.floor((box[2] - box[0]) * tWidth);
          const cropHeight = Math.floor((box[3] - box[1]) * tHeight);
          const startPos = [tStartY, tStartX, 0];
          const cropSize = [cropHeight, cropLength, 3];
          const cropped = tf.slice(tensor, startPos, cropSize);

          tf.browser.toPixels(cropped, cropRef.current);
        });
      });
    });
  }, [imgRef, canvasRef, cropRef]);

  return (
    <>
      <h1>Inception</h1>

      <p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/animal-gf6f46e288_640.jpg" alt="truck" ref={imgRef} />
        <br />
        <a
          href="https://pixabay.com/photos/468228/"
          target="_blank"
          rel="noopener noreferrer"
        >
          https://pixabay.com/photos/468228/
        </a>
      </p>

      <p>
        <canvas ref={canvasRef} width={640} height={425} />
      </p>

      <p>
        <canvas ref={cropRef} width={300} height={300} />
      </p>
    </>
  );
}
