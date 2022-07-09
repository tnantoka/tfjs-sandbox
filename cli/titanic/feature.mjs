import * as tf from '@tensorflow/tfjs';
import * as dfd from 'danfojs-node';
import _ from 'lodash';

const df = await dfd.readCSV('data/cleaned/newTrain.csv');
const dft = await dfd.readCSV('data/cleaned/newTest.csv');
const mega = dfd.concat({ dfList: [df, dft], axis: 0 });

const sexOneHot = dfd.getDummies(mega['Sex']);

mega.drop({ columns: ['Sex'], axis: 1, inplace: true });
mega.addColumn('male', sexOneHot[sexOneHot.columns[0]], { inplace: true });
mega.addColumn('female', sexOneHot[sexOneHot.columns[1]], { inplace: true });

const ageToBucket = (age) => {
  if (age < 10) {
    return 0;
  } else if (age < 40) {
    return 1;
  } else {
    return 2;
  }
}

const ageBuckets = mega['Age'].apply(ageToBucket);
mega.addColumn('Age_bucket', ageBuckets, { inplace: true });

const scaler = new dfd.MinMaxScaler();
scaler.fit(mega);
const scaledData = scaler.transform(mega)

await dfd.toCSV(scaledData, { filePath: 'data/featured/titanic.csv' });
