import * as tf from '@tensorflow/tfjs';
import * as dfd from 'danfojs-node';
import _ from 'lodash';

const df = await dfd.readCSV('data/featured/titanic.csv');

const trainX = df.iloc({ columns: ['1:'] }).tensor;
const trainY = df['Survived'].tensor;

const inputShape = [trainX.shape[1]]

const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape,
    units: 120,
    activation: 'relu',
    kernelInitializer: 'heNormal',
  })
);
model.add(
  tf.layers.dense({
    units: 64,
    activation: 'relu',
  })
);
model.add(
  tf.layers.dense({
    units: 32,
    activation: 'relu',
  })
);
model.add(
  tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
  })
);

model.compile({
  optimizer: 'adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

await model.fit(trainX, trainY, {
  batchSize: 32,
  epochs: 100,  
  validationSplit: 0.2,
});
