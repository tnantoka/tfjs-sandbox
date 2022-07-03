import React from 'react';
import * as tf from '@tensorflow/tfjs';

const empty = new Array(9).fill(0);
const block = [-1, 0, 0, 1, 1, -1, 0, 0, -1];
const kill = [1, 0, 1, 0, -1, -1, -1, 0, 1];

type BoardProps = {
  cells: number[];
  player: string;
  bot: string;
  next?: number;
};

const Board: React.FC<BoardProps> = ({ cells, player, bot, next = -1 }) => {
  return (
    <table>
      <tbody>
        {new Array(3).fill(0).map((_, i) => (
          <tr key={i}>
            {new Array(3).fill(0).map((_, j) => {
              const index = i * 3 + j;
              return (
                <td key={j}>
                  {index === next
                    ? '*'
                    : cells[index] === 0
                    ? '\u00A0\u00A0'
                    : cells[index] === 1
                    ? bot
                    : player}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default function TicTacToe() {
  const [nexts, setNexts] = React.useState([]);

  React.useEffect(() => {
    tf.ready().then(() => {
      const modelPath = '/model/ttt_model.json';
      tf.tidy(() => {
        tf.loadLayersModel(modelPath).then((model) => {
          const emptyBoard = tf.tensor(empty);
          const betterBlockMe = tf.tensor(block);
          const goForTheKill = tf.tensor(kill);

          const matches = tf.stack([emptyBoard, betterBlockMe, goForTheKill]);
          const result = model.predict(matches);
          setNexts(tf.topk(result, 1).indices.dataSync());
        });
      });
    });
  });

  return (
    <>
      <h1>Tic Tac Toe</h1>

      <h2>Empty</h2>
      <h3>Before</h3>
      <Board cells={empty} player="o" bot="x" />
      <h3>After</h3>
      <Board cells={empty} player="o" bot="x" next={nexts[0]} />

      <h2>Block</h2>
      <h3>Before</h3>
      <Board cells={block} player="x" bot="o" />
      <h3>After</h3>
      <Board cells={block} player="x" bot="o" next={nexts[1]} />

      <h2>Kill</h2>
      <h3>Before</h3>
      <Board cells={kill} player="o" bot="x" />
      <h3>After</h3>
      <Board cells={kill} player="o" bot="x" next={nexts[2]} />
    </>
  );
}
