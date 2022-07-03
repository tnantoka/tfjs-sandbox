import React from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Tensor() {
  const [results, setResults] = React.useState('');

  React.useEffect(() => {
    let results = [];
    const data = [1, 5, 3, 4, 2];
    results.push('// tensor');
    results.push(JSON.stringify(tf.tensor(data), null, 2));

    results.push('\n// tensor1d(int32)');
    results.push(JSON.stringify(tf.tensor1d(data, 'int32'), null, 2));

    results.push('\n// array');
    results.push(JSON.stringify(tf.tensor1d(data).arraySync(), null, 2));

    results.push('\n// data');
    results.push(JSON.stringify(tf.tensor1d(data).dataSync(), null, 2));

    const mat1 = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const mat2 = [
      [10, 11, 12],
      [13, 14, 15],
      [16, 17, 18],
    ];

    results.push('\n// mat');
    results.push(JSON.stringify(tf.matMul(mat1, mat2).arraySync(), null, 2));

    setResults(results.join('\n'));

    console.info('start', tf.memory().numTensors);
    let keeper, chaser, speeker, beater;
    tf.tidy(() => {
      keeper = tf.tensor(data);
      chaser = tf.tensor(data);
      speeker = tf.tensor(data);
      beater = tf.tensor(data);

      console.info('inside tidy', tf.memory().numTensors);

      tf.keep(keeper);

      return chaser;
    });
    console.info('after tidy', tf.memory().numTensors);

    keeper.dispose();
    chaser.dispose();

    console.log('end', tf.memory().numTensors);
  }, []);

  return (
    <>
      <h1>Tensor</h1>
      <pre>{results}</pre>
    </>
  );
}
