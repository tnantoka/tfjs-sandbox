import React from 'react';
import * as tf from '@tensorflow/tfjs';

import { INCEPTION_CLASSES } from './inception_labels';

export default function Inception() {
  const imgRef = React.useRef(null);
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
  }, [imgRef]);

  return (
    <>
      <h1>Inception</h1>

      <p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/transport-g25d4a18f6_640.jpg" alt="truck" ref={imgRef} />
        <br />
        <a
          href="https://pixabay.com/photos/3369756/"
          target="_blank"
          rel="noopener noreferrer"
        >
          https://pixabay.com/photos/3369756/
        </a>
      </p>
      <ol>
        {Object.values(predictions).map((prediction) => (
          <li key={prediction}>{INCEPTION_CLASSES[prediction]}</li>
        ))}
      </ol>
    </>
  );
}
