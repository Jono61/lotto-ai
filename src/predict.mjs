import * as tf from '@tensorflow/tfjs-node';
import { prepareData, loadData, decodePredictionsTemp, decodeBestBet } from "./lottoflow.mjs"

async function predict(data) {

    const {
        inputSequences,
    } = prepareData(data);

    const xs = tf.tensor4d(inputSequences);

    // Load the model
    const model = await tf.loadLayersModel('file://./lotto_model/model.json');

    const trainSize = Math.floor(0.80 * xs.shape[0]);
   
     // 4. Validation / Prediction
     // Use the remaining 20% of the data for validation / prediction
     const xsVal = xs.slice([trainSize, 0, 0, 0], [-1, -1, -1, -1]);
     const predictions = model.predict(xsVal);
     
     console.log("Prediction shape:", predictions.shape);
     predictions.print();

     const decodedPredictionsTemp = decodePredictionsTemp(predictions);
     console.log("Decoded predictions (temp .99):", decodedPredictionsTemp);

     const bestBet = decodeBestBet(predictions);
     console.log("Best bet:", bestBet);
}

predict(await loadData());