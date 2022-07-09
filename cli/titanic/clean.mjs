import * as tf from '@tensorflow/tfjs';
import * as dfd from 'danfojs-node';
import _ from 'lodash';

console.log(tf.version.tfjs);

// const df = await dfd.readCSV('titanic_data/train.csv');
// df.head().print();
// df.describe().print();

// const emptySpots = df.isNa().sum({ axis: 0 });
// emptySpots.print();
// const emptyRate = emptySpots.div(df.isNa().count({ axis: 0 }));
// emptyRate.print();
// 
// const df2 = df.loc({ columns: df.columns.slice(10) })
// const emptySpots2 = df2.isNa().sum({ axis: 0 });
// emptySpots2.print();
// const emptyRate2 = emptySpots2.div(df2.isNa().count({ axis: 0 }));
// emptyRate2.print();

const df = await dfd.readCSV('data/train.csv');
const dft = await dfd.readCSV('data/test.csv');
const mega = dfd.concat({ dfList: [df, dft], axis: 0 });

const clean = mega.drop({ columns: ['Name', 'PassengerId', 'Ticket', 'Cabin'] })
const onlyFull = clean.dropNa();
console.log(`shape ${onlyFull.shape[0]}`);

const encode = new dfd.LabelEncoder();
encode.fit(onlyFull['Embarked']);
onlyFull['Embarked'] = encode.transform(onlyFull['Embarked'].values);

encode.fit(onlyFull['Sex'])
onlyFull['Sex'] = encode.transform(onlyFull['Sex'].values)

onlyFull.resetIndex({ inplace: true });
onlyFull.head().print()

const index = _.shuffle(onlyFull.index).slice(0, 800);

const newTrain = onlyFull.iloc({ rows: index });
const newTest = onlyFull.drop({ index });

await dfd.toCSV(newTrain, { filePath: 'data/cleaned/newTrain.csv' });
console.log(newTrain.shape[0]);
await dfd.toCSV(newTest, { filePath: 'data/cleaned/newTest.csv' });
console.log(newTest.shape[0]);
