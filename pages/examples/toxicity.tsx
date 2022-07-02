import React from 'react';
import * as toxicity from '@tensorflow-models/toxicity';

const threshold = 0.5;

const ja = {
  identity_attack: '名指しの攻撃',
  insult: '侮辱',
  obscene: '猥褻',
  severe_toxicity: '極端に有害',
  sexual_explicit: 'あからさまに性的な内容',
  threat: '脅迫',
  toxicity: '有害',
}

export default function Toxicity() {
  const [sentences, setSentences] = React.useState(
`You are a poopy head!
I like turtles
Shut up!
`);
  const [predictions, setPredictions] = React.useState([]);

  const inputs = sentences.split('\n').filter((sentence) => sentence);

  React.useEffect(() => {
    toxicity.load(threshold).then((model) => {
      model.classify(inputs).then((predictions) => {
        console.info(predictions);
        setPredictions(predictions);
      });
    });
  }, [sentences])

  const onChangeSentences = React.useCallback((e) => {
    setSentences(e.target.value);
  }, [])

  return (
    <>
      <h1>Toxicity</h1>
      <p>
        <textarea
          value={sentences}
          onChange={onChangeSentences}
          cols={50}
          rows={5}
        />
      </p>
      <table>
        <thead>
          <tr>
            <th></th>
            {predictions.map(({ label }) => (
              <th key={label}>{ja[label]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {inputs.map((text, i) => (
            <tr key={text}>
              <td>{text}</td>
              {predictions.map(({ label, results }) => (
                results[i] !== undefined && (
                  <td key={label}>{Math.round(results[i].probabilities[1] * 100)}%</td>
                )
              ))}
            </tr> 
          ))}
        </tbody>
      </table>
      
    </>
  );
}
