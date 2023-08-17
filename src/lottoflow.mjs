import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

export function oneHotEncode(sequence, range) {
    return sequence.map(num => {
        const encoded = new Array(range).fill(0);
        encoded[num - 1] = 1;
        return encoded;
    });
}

function attentionMechanism(dropoutOutput) {
    const attentionDense = tf.layers.dense({ units: 1, activation: 'tanh' });
    const attentionWeights = attentionDense.apply(dropoutOutput);
    const contextVector = tf.layers.dot({ axes: [1, 1] }).apply([attentionWeights, dropoutOutput]);
    return contextVector;
}

function buildALSTMModel(inputShape, LSTMUnits, outputUnits) {
    const inputs = tf.input({ shape: inputShape });
    const reshaped = tf.layers.reshape({ targetShape: [inputShape[0], inputShape[1] * inputShape[2]] }).apply(inputs);
    const LSTMOutput = tf.layers.lstm({
        units: LSTMUnits,
        returnSequences: true,
        kernelInitializer: 'glorotNormal',
        recurrentInitializer: 'glorotNormal',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) // L2 Regularization
    }).apply(reshaped);
    const dropoutLayer = tf.layers.dropout({ rate: 0.5 }).apply(LSTMOutput);  // Dropout Layer
    const contextVector = attentionMechanism(dropoutLayer);
    const flattened = tf.layers.flatten().apply(contextVector);
    const outputs = tf.layers.dense({
        units: outputUnits,
        activation: 'sigmoid',
        kernelInitializer: 'glorotNormal',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) // L2 Regularization
    }).apply(flattened);
    const model = tf.model({ inputs, outputs });
    console.log("Model summary:");
    model.summary();
    return model;
}

export function prepareData(data) {
    const sequences = data.map(entry => [1,2,3,4,5,6].map(i => entry[`Zahl${i}`]));
    const encodedSequences = sequences.map(seq => oneHotEncode(seq, 49));
    const encodedSuperzahlen = data.map(entry => oneHotEncode([entry['Superzahl']], 10)[0]);

    const inputSequences = [];
    const outputSequences = [];

    if(encodedSequences.length >= 6) {
        for (let i = 0; i < encodedSequences.length - 5; i++) {
            const inputWindow = encodedSequences.slice(i, i + 5).map((seq, idx) => {
                // Extend encodedSuperzahlen[i + idx] to be of length 49
                const extendedSuperzahl = [...encodedSuperzahlen[i + idx], ...new Array(39).fill(0)];

                // Add extendedSuperzahl as a new "row" in the sequence
                return [...seq, extendedSuperzahl];
            });
            inputSequences.push(inputWindow);

            const flattenedNumbers = [].concat(...encodedSequences[i + 5]);
            outputSequences.push([...flattenedNumbers, ...encodedSuperzahlen[i + 5]]);
        }
    }

    // Added check
    if (inputSequences.length !== outputSequences.length) {
        console.error("Mismatch in input and output sequences length");
        process.exit(1);
    }

    inputSequences.forEach((seq, idx) => {
        if (seq.length !== 5 || seq[0].length !== 7 || seq[0][0].length !== 49) {
            console.log("Mismatch in inputSequences at index", idx);
        }
    });

    return {
        inputSequences,
        outputSequences,
    };
}

export function decodeBestBet(predictions) {
    const numbers = [];
    for (let i = 0; i < 6; i++) {
        const startIdx = i * 49;
        const number = tf.argMax(predictions.slice([0, startIdx], [-1, 49]), 1).add(1).arraySync();
        numbers.push(number);
    }

    // Get the index of the maximum value in the last 10 values (1 "Superzahl")
    const superzahl = tf.argMax(predictions.slice([0, 294], [-1, 10]), 1).arraySync();

    const decodedPredictions = numbers[0].map((_, idx) => {
        return {
            Zahlen: numbers.map(num => num[idx]).sort((a, b) => a - b),
            Superzahl: superzahl[idx]
        };
    });

    return decodedPredictions[0];
}

export const train = async(data) => {
    // 1. Data Preprocessing
    const {
        inputSequences,
        outputSequences,
    } = prepareData(data);

    const xs = tf.tensor4d(inputSequences);
    const ys = tf.tensor2d(outputSequences);

    console.log("Input tensor shape (xs):", xs.shape);
    console.log("Output tensor shape (ys):", ys.shape);

    // 2. Model Definition
    const model = buildALSTMModel([5, 7, 49], 128, 304);
    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    
     // Use 80% of the data for training
     const trainSize = Math.floor(0.80 * xs.shape[0]);
     const xsTrain = xs.slice([0, 0, 0, 0], [trainSize, -1, -1, -1]);
     const ysTrain = ys.slice([0, 0], [trainSize, -1]);
     
     await model.fit(xsTrain, ysTrain, { 
         epochs: 100,
         batchSize: 8,
         shuffle: true,
         validationSplit: 0.2,
         verbose: 1
     });    
     
     // 4. Validation / Prediction
     // Use the remaining 20% of the data for validation / prediction
     const xsVal = xs.slice([trainSize, 0, 0, 0], [-1, -1, -1, -1]);
     const predictions = model.predict(xsVal);
     
     console.log("Prediction shape:", predictions.shape);
     predictions.print();

     const decodedPredictions = decodeBestBet(predictions);
     console.log("Decoded predictions:", decodedPredictions);

    // 6. Saving the model
    await model.save('file://./lotto_model');
};

export function decodePredictionsTemp(predictions, temperature = 0.99) {
    const numbers = [];

    for (let i = 0; i < 6; i++) {
        const startIdx = i * 49;
        const scaledProbs = predictions.slice([0, startIdx], [-1, 49]).div(tf.scalar(temperature));
        let number;
        let unique = false;

        while (!unique) {
            number = tf.multinomial(scaledProbs.softmax(1), 1).add(1).arraySync();
            unique = hasDuplicates(number[0]) === false;
        }
        
        numbers.push(number);
    }

    // Get the index of the maximum value in the last 10 values for "Superzahl"
    const superzahlProbs = predictions.slice([0, 294], [-1, 10]).div(tf.scalar(temperature));
    const superzahl = tf.multinomial(superzahlProbs.softmax(1), 1).arraySync();

    const decodedPredictions = numbers[0].map((_, idx) => {
        return {
            Zahlen: numbers.map(num => num[idx][0]).sort((a, b) => a - b),
            Superzahl: superzahl[idx][0]
        };
    });

    // Filter to remove duplicate "Zahlen" sequences
    const uniquePredictions = removeDuplicateZahlen(decodedPredictions);

    return uniquePredictions;
}

// Helper function to check for duplicate values in an array
function hasDuplicates(array) {
    return (new Set(array)).size !== array.length;
}

// Helper function to remove duplicate "Zahlen" sequences
function removeDuplicateZahlen(predictions) {
    const seen = new Set();
    return predictions.filter(prediction => {
        const strRep = JSON.stringify(prediction.Zahlen);

        let hasDuplicateSingleNumber = false;
        prediction.Zahlen.forEach((num, i) => {
            if (prediction.Zahlen[i+1] && prediction.Zahlen[i+1] === num) {
                hasDuplicateSingleNumber = true;
            }
        });

        if (hasDuplicateSingleNumber) {
            return false;
        }

        if (seen.has(strRep)) {
            return false;
        } else {
            seen.add(strRep);
            return true;
        }
    });
}

export async function loadData() {
    return new Promise((resolve) => {
        fs.readFile('./data/lotto.json', 'utf8', async (err, jsonString) => {
            if (err) {
                console.log("File read failed:", err);
                return;
            }
            resolve(JSON.parse(jsonString))
        });
    });
}
