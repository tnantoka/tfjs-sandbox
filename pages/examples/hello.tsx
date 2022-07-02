import * as tf from '@tensorflow/tfjs';

export default function Hello() {
  return (
    <>
      <h1>Hello</h1>
      <p>{tf.version.tfjs}</p>
    </>
  );
}
