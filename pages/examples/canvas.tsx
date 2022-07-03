import React from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Canvas() {
  const checkRef = React.useRef(null);
  const randomRef = React.useRef(null);
  const imgRef = React.useRef(null);
  const mirrorRef = React.useRef(null);
  const resizeRef = React.useRef(null);
  const cropRef = React.useRef(null);
  const sortRef = React.useRef(null);
  const [pixels, setPixels] = React.useState('');

  React.useEffect(() => {
    tf.tidy(() => {
      const lil = tf.tensor([
        [[1], [0]],
        [[0], [1]],
      ]);
      const big = lil.tile([
        checkRef.current.height,
        checkRef.current.width,
        1,
      ]);
      tf.browser.toPixels(big, checkRef.current);
    });
  }, [checkRef]);

  React.useEffect(() => {
    tf.tidy(() => {
      const bigMess = tf.randomUniform([200, 200, 3]);
      tf.browser.toPixels(bigMess, randomRef.current);
    });
  }, [randomRef]);

  React.useEffect(() => {
    if (
      !imgRef.current ||
      !mirrorRef.current ||
      !resizeRef.current ||
      !cropRef.current
    ) {
      return;
    }

    tf.tidy(() => {
      const tensor = tf.browser.fromPixels(imgRef.current);
      setPixels(JSON.stringify(tensor.shape));

      // const reversed = tf.reverse(tensor, 1);
      const batch = tf.expandDims(tensor.asType('float32'));
      const reversed = tf.squeeze(
        tf.image.flipLeftRight(batch).asType('int32')
      );
      tf.browser.toPixels(reversed, mirrorRef.current);

      const resized = tf.image
        .resizeBilinear(tensor, [80, 80], true)
        .asType('int32');
      tf.browser.toPixels(resized, resizeRef.current);

      const cropped = tf.slice(tensor, [0, 10, 0], [30, 20]);
      tf.browser.toPixels(cropped, cropRef.current);
    });
  }, [imgRef, mirrorRef, resizeRef]);

  React.useEffect(() => {
    tf.tidy(() => {
      const rando = tf.randomUniform([200, 200]);
      const sorted = tf.topk(rando, 200).values;
      const reshaped = sorted.reshape([200, 200, 1]);
      tf.browser.toPixels(reshaped, sortRef.current);
    });
  }, [sortRef]);

  return (
    <>
      <h1>Canvas</h1>
      <h2>Check</h2>
      <canvas
        ref={checkRef}
        width={10}
        height={5}
        style={{ width: 400, height: 200, imageRendering: 'pixelated' }}
      />
      <h2>Random</h2>
      <canvas
        ref={randomRef}
        width={200}
        height={200}
        style={{ imageRendering: 'pixelated' }}
      />
      <h2>fromPixels</h2>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img ref={imgRef} src="/check.png" alt="check" />
      <pre>{pixels}</pre>
      <h2>Mirror</h2>
      <canvas
        ref={mirrorRef}
        width={40}
        height={40}
        style={{ imageRendering: 'pixelated' }}
      />
      <h2>Resize</h2>
      <canvas
        ref={resizeRef}
        width={80}
        height={80}
        style={{ imageRendering: 'pixelated' }}
      />
      <h2>Crop</h2>
      <canvas
        ref={cropRef}
        width={20}
        height={30}
        style={{ imageRendering: 'pixelated' }}
      />
      <h2>Sort</h2>
      <canvas
        ref={sortRef}
        width={200}
        height={200}
        style={{ imageRendering: 'pixelated' }}
      />
    </>
  );
}
