// @flow

type Model = {
  weights: Array<number>,
  bias: number
};

const setW0 = (w0, model): Model => ({ weights: [ w0, model.weights[1] ], bias: model.bias });
const setW1 = (w1, model): Model => ({ weights: [ model.weights[0], w1 ], bias: model.bias });
const setB = (b, model): Model => ({ weights: model.weights, bias: b });

type Data = Array<{ input: Array<number>, expectedOutput: number}>;

const makePerceptron = ({ weights, bias }: Model) => (input: Array<number>): number => {
  let acc = 0;
  for (let i = 0; i < weights.length; i += 1) {
    acc += weights[i] * input[i];
  }
  return 1 / (1 + Math.exp(-1 * (acc + bias)));
};

const generatePerceptronModel = () => {
  const randomBound = () => 1 - (Math.random() * 2)
  const weights = [ randomBound(), randomBound() ];
  const bias = 0;
  return { weights, bias };
};

const numericalDerivative = (f: number => number) => (x0: number): number => {
  const delta = 0.0001;
  const nd = (f(x0 + delta) - f(x0)) / delta
  return nd
};


// Stochastic Gradient Descent with batch size of 1
const train0: Data => Model => Model = (() => {
  const getCost = (input: Array<number>, expectedOutput: number) => (model: Model): number => {
    const output = makePerceptron(model)(input);
    return Math.pow(output - expectedOutput, 2); // squared error
  };

  const step = (input: Array<number>, expectedOutput: number) => (model: Model): { cost: number, nextModel: Model } => {
    const _getCost = getCost(input, expectedOutput);
    const cost = _getCost(model); // FIXME: optimize later

    const dw0 = numericalDerivative((w0) => _getCost(setW0(w0, model)))(model.weights[0]);
    const dw1 = numericalDerivative((w1) => _getCost(setW1(w1, model)))(model.weights[1]);
    const db = numericalDerivative((b) => _getCost(setB(b, model)))(model.bias);

    const learningRate = 0.5;

    const nextModel = {
      weights: [
        model.weights[0] - (dw0 * learningRate),
        model.weights[1] - (dw1 * learningRate),
      ],
      bias: model.bias - (db * learningRate),
    };

    return {
      cost,
      nextModel,
    };
  };

  const train = (data: Data) => (model: Model): Model=> {
    let m = model;
    for (let i = 0; i < data.length; i += 1 ) {
      const { input, expectedOutput } = data[i];
      const { cost, nextModel } = step(input, expectedOutput)(m);
      m = nextModel;
    }
    return m
  };

  return train;
})();

// Gradient descent
const train1: { steps: number } => Data => Model => Model = (() => {

  // mean squared error
  const getCost = (data: Data) => (model: Model): number => {
    const perceptron = makePerceptron(model);
    let acc = 0;
    for (let i = 0; i < data.length; i += 1) {
      const { input, expectedOutput } = data[i];
      const output = perceptron(input);
      acc += Math.pow(output - expectedOutput, 2);
    }
    return acc / data.length;
  };

  const step = (data: Data) => (model: Model): { cost: number, nextModel: Model } => {
    const _getCost = getCost(data);
    const cost = _getCost(model); // FIXME: optimize later

    const dw0 = numericalDerivative((w0) => _getCost(setW0(w0, model)))(model.weights[0]);
    const dw1 = numericalDerivative((w1) => _getCost(setW1(w1, model)))(model.weights[1]);
    const db = numericalDerivative((b) => _getCost(setB(b, model)))(model.bias);

    const learningRate = 0.5;

    const nextModel = {
      weights: [
        model.weights[0] - (dw0 * learningRate),
        model.weights[1] - (dw1 * learningRate),
      ],
      bias: model.bias - (db * learningRate),
    };

    return {
      cost,
      nextModel,
    };
  };

  const train = (parameters: { steps: number }) => (data: Data) => (model: Model): Model=> {
    const { steps } = parameters;
    let m = model;
    for (let i = 0; i < steps; i += 1 ) {
      const { cost, nextModel } = step(data)(m);
      m = nextModel;
    }
    return m
  };

  return train;
})();

const sampleRandomly = <A>(n: number, array: Array<A>): Array<A> => {
  const ret = [];
  for (let i = 0; i < n; i += 1) {
    const randIndex = Math.floor(Math.random() * array.length);
    ret.push(array[randIndex]);
  }
  return ret;
};

// Stochastic Gradient descent
const train2: { steps: number, batchSize: number } => Data => Model => Model = (() => {

  // mean squared error
  const getCost = (data: Data) => (model: Model): number => {
    const perceptron = makePerceptron(model);
    let acc = 0;
    for (let i = 0; i < data.length; i += 1) {
      const { input, expectedOutput } = data[i];
      const output = perceptron(input);
      acc += Math.pow(output - expectedOutput, 2);
    }
    return acc / data.length;
  };

  const step = (data: Data) => (model: Model): { cost: number, nextModel: Model } => {
    const _getCost = getCost(data);
    const cost = _getCost(model); // FIXME: optimize later

    const dw0 = numericalDerivative((w0) => _getCost(setW0(w0, model)))(model.weights[0]);
    const dw1 = numericalDerivative((w1) => _getCost(setW1(w1, model)))(model.weights[1]);
    const db = numericalDerivative((b) => _getCost(setB(b, model)))(model.bias);

    const learningRate = 0.5;

    const nextModel = {
      weights: [
        model.weights[0] - (dw0 * learningRate),
        model.weights[1] - (dw1 * learningRate),
      ],
      bias: model.bias - (db * learningRate),
    };

    return {
      cost,
      nextModel,
    };
  };

  const train = (parameters: { steps: number, batchSize: number }) => (data: Data) => (model: Model): Model=> {
    const { steps, batchSize } = parameters;
    let m = model;
    for (let i = 0; i < steps; i += 1 ) {
      const sample = sampleRandomly(batchSize, data);
      const { cost, nextModel } = step(sample)(m);
      m = nextModel;
    }
    return m
  };

  return train;
})();


module.exports = {
  makePerceptron,
  generatePerceptronModel,
  train0,
  train1,
  train2,
};
