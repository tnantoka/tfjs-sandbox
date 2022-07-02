import React from 'react';
import Head from 'next/head';

export default function Mobilenet() {
  const [predictions, setPredictions] = React.useState([]);

  React.useLayoutEffect(() => {
    const img = document.querySelector('img');
    const { mobilenet } = window;

    if (img && mobilenet) {
      mobilenet.load().then((model) => {
        model.classify(img).then((predictions) => {
          console.info(predictions);
          setPredictions(predictions);
        });
      });
    }
  }, []);

  return (
    <>
      <Head>
        {/* eslint-disable-next-line @next/next/no-sync-scripts */}
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.1" />
        {/* eslint-disable-next-line @next/next/no-sync-scripts */}
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0" />
      </Head>

      <h1>Truck</h1>

      <p>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/transport-g25d4a18f6_640.jpg" alt="truck" />
        <br />
        <a
          href="https://pixabay.com/photos/3369756/"
          target="_blank"
          rel="noopener noreferrer"
        >
          https://pixabay.com/photos/3369756/
        </a>
      </p>
      <table>
        <tbody>
          {predictions.map(({ className, probability }) => (
            <tr key={className}>
              <th>{className}</th>
              <td>{Math.round(probability * 100)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
