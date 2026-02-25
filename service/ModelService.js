import tf from "@tensorflow/tfjs";

class ModelService {
  static trainModel = async (inputXs, outputYs) => {
    const model = tf.sequential();

    model.add(
      tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
    );

    model.add(tf.layers.dense({ units: 3, activation: "softmax" }));
    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    await model.fit(inputXs, outputYs, {
      verbose: 0,
      epochs: 100,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, log) =>
          console.log(`Epoch: ${epoch}: loss = ${log.loss}`),
      },
    });

    return model;
  };

  static predict = async (model, personTensor) => {
    const tfInput = tf.tensor2d(personTensor);

    const pred = model.predict(tfInput);
    const predArray = await pred.array();

    return predArray[0].map((prob, index) => ({
      prob,
      index,
    }));
  };
}

export default ModelService;
