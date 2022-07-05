import React from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Train() {
  const [logs, setLogs] = React.useState([]);
  const [answer, setAnswer] = React.useState(0);
  const isTraining = React.useRef(false);

  React.useEffect(() => {
    if (isTraining.current) {
      return;
    }
    (async () => {
      isTraining.current = true;

      await tf.ready();

      const rawXs = [];
      const rawYs = [];

      const dataSize = 10;
      const stepSize = 0.001;
      for (let i = 0; i < dataSize; i += stepSize) {
        rawXs.push(i);
        rawYs.push(i * i);
      }

      const xs = tf.tensor(rawXs);
      const ys = tf.tensor(rawYs);

      let logs = [];
      const callbacks = {
        onEpochEnd: (epoch, log) => {
          logs.push(log);
          console.log(epoch, log);
        },
      };

      const model = tf.sequential();
      model.add(
        tf.layers.dense({
          inputShape: 1,
          units: 20,
          activation: 'relu',
        })
      );
      model.add(
        tf.layers.dense({
          units: 20,
          activation: 'relu',
        })
      );
      model.add(
        tf.layers.dense({
          units: 1,
        })
      );

      model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
      });

      console.time('Train');
      await model.fit(xs, ys, {
        epochs: 50,
        callbacks,
        batchSize: 64,
      });
      console.timeEnd('Train');

      const next = tf.tensor([7]);
      const answer = model.predict(next);
      setAnswer(answer.dataSync()[0]);

      answer.dispose();
      xs.dispose();
      ys.dispose();
      model.dispose;

      setLogs(logs);

      isTraining.current = false;
    })();
  }, []);

  return (
    <>
      <h1>Train</h1>
      <p>7 * 7 = {answer}</p>
      <ul>
        {logs.map((log, i) => (
          <li key={i}>
            {i + 1}: {log.loss}
          </li>
        ))}
      </ul>
    </>
  );
}
