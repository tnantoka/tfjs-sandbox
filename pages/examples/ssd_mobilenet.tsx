import React from 'react';
import * as tf from '@tensorflow/tfjs';

import { CLASSES } from './ssd_mobilenet_labels';

export default function SsdMobilenet() {
  const imgRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const [predictions, setPredictions] = React.useState([]);

  React.useEffect(() => {
    (async () => {
      await tf.ready();
      const modelPath =
        'https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1';

      const model = await tf.loadGraphModel(modelPath, { fromTFHub: true });
      const tensor = tf.browser.fromPixels(imgRef.current);
      const redified = tf.expandDims(tensor, 0);
      const results = await model.executeAsync(redified);

      const detectionThreshold = 0.4;
      const iouThreshold = 0.5;
      const maxBoxes = 20;
      const prominentDetection = tf.topk(results[0]);
      const justBoxes = results[1].squeeze();
      const justValues = prominentDetection.values.squeeze();

      const imgWidth = imgRef.current.width;
      const imgHeight = imgRef.current.height;
      canvasRef.current.width = imgWidth;
      canvasRef.current.height = imgHeight;

      const [maxIndices, scores, boxes] = await Promise.all([
        prominentDetection.indices.data(),
        justValues.array(),
        justBoxes.array(),
      ]);

      const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
        justBoxes,
        justValues,
        maxBoxes,
        iouThreshold,
        detectionThreshold,
        1
      );

      const chosen = await nmsDetections.selectedIndices.data();

      tf.dispose([
        results[0],
        results[1],
        model,
        nmsDetections.selectedIndices,
        nmsDetections.selectedScores,
        prominentDetection.indices,
        prominentDetection.values,
        tensor,
        redified,
        justBoxes,
        justValues,
      ]);

      const ctx = canvasRef.current.getContext('2d');
      ctx.font = '16px sans-serif';
      ctx.textBaseline = 'top';

      chosen.forEach((detection) => {
        ctx.strokeStyle = '#0F0';
        ctx.lineWidth = 4;
        ctx.globalCompositeOperation = 'destination-over';
        const detectedIndex = maxIndices[detection];
        const detectedClass = CLASSES[detectedIndex];
        const detectedScore = scores[detection];
        const dBox = boxes[detection];

        const startY = Math.max(dBox[0] * imgHeight, 0);
        const startX = Math.max(dBox[1] * imgWidth, 0);
        const height = (dBox[2] - dBox[0]) * imgHeight;
        const width = (dBox[3] - dBox[1]) * imgWidth;
        ctx.strokeRect(startX, startY, width, height);

        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = '#0B0';
        const textHeight = 16;
        const textPad = 4;
        const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(startX, startY, textWidth + textPad, textHeight + textPad);

        ctx.fillStyle = '#000000';
        ctx.fillText(label, startX, startY);
      });

      console.log('Tensor Memory Status:', tf.memory().numTensors);
    })();
  }, [imgRef, canvasRef]);

  return (
    <>
      <h1>Inception</h1>

      <p style={{ position: 'relative' }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/pizza-g7028159ed_640.jpg" alt="truck" ref={imgRef} />
        <canvas
          ref={canvasRef}
          style={{ position: 'absolute', left: 0, top: 0 }}
        />
      </p>

      <p>
        <a
          href="https://pixabay.com/photos/2000615/"
          target="_blank"
          rel="noopener noreferrer"
        >
          https://pixabay.com/photos/2000615/
        </a>
      </p>
    </>
  );
}
