import React from 'react';
import * as tf from '@tensorflow/tfjs';

const users = ['Gant', 'Todd', 'Jed', 'Justin'];
const bands = [
  'Nirvana',
  'Nine Inch Nails',
  'Backstreet Boys',
  'N Sync',
  'Night Club',
  'Apashe',
  'STP',
];
const features = [
  'Grunge',
  'Rock',
  'Industrial',
  'Boy Band',
  'Dance',
  'Techno',
];

const userVotes = tf.tensor([
  [10, 9, 1, 1, 8, 7, 8],
  [6, 8, 2, 2, 0, 10, 0],
  [0, 2, 10, 9, 3, 7, 0],
  [7, 4, 2, 3, 6, 5, 5],
]);

const bandFeats = tf.tensor([
  [1, 1, 0, 0, 0, 0],
  [1, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0, 1],
  [0, 0, 1, 0, 0, 1],
  [1, 1, 0, 0, 0, 0],
]);

export default function Recommendation() {
  const [topGenres, setTopGenres] = React.useState([]);
  React.useEffect(() => {
    const userFeats = tf.matMul(userVotes, bandFeats);
    const topUserFeats = tf.topk(userFeats, features.length);
    const topGenres = topUserFeats.indices.arraySync();
    setTopGenres(topGenres);
  }, []);

  return (
    <>
      <h1>Recommendation</h1>
      <h2>Users</h2>
      <table>
        <thead>
          <tr>
            <th></th>
            {bands.map((band) => (
              <th key={band}>{band}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {users.map((user, i) => (
            <tr key={user}>
              <td>{user}</td>
              {userVotes.arraySync()[i].map((vote, i) => (
                <td key={i}>{vote}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <h2>Bands</h2>
      <table>
        <thead>
          <tr>
            <th></th>
            {features.map((feature) => (
              <th key={feature}>{feature}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bands.map((band, i) => (
            <tr key={band}>
              <td>{band}</td>
              {bandFeats.arraySync()[i].map((feat, i) => (
                <td key={i}>{feat}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <h2>Recommendations</h2>
      <table>
        <tbody>
          {users.map((user, i) => (
            <tr key={user}>
              <th>{user}</th>
              {topGenres.length > 0 &&
                topGenres[i].map((genre) => (
                  <td key={genre}>{features[genre]}</td>
                ))}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
