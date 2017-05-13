// @flow

type Model = {
  weights: Array<number>,
  bias: number
};

const setW0 = (w0, model): Model => ({ weights: [ w0, model.weights[1] ], bias: model.bias });
const setW1 = (w1, model): Model => ({ weights: [ model.weights[0], w1 ], bias: model.bias });
const setB = (b, model): Model => ({ weights: model.weights, bias: b });

type Data = Array<{ input: Array<number>, expectedOutput: number}>;

const sigmoid = (x) => 1 / (1 + Math.exp(-1 * x));
const sigmoidPrime = (x) => {
  const s = sigmoid(x);
  return s * (1 - s);
};

const gradPerceptron = (model: Model, input: Array<number>) => {
  const { weights, bias } = model;
  const foo = sigmoidPrime(dotProduct(weights, input) + bias);
  return {
    dp_dw0: foo * input[0],
    dp_dw1: foo * input[1],
    dp_db: foo,
  };
};

const dotProduct = (x: Array<number>, y: Array<number>) => {
  let acc = 0;
  for (let i = 0; i < x.length; i += 1) {
    acc += x[i] * y[i];
  }
  return acc;
};

const makePerceptron = ({ weights, bias }: Model) => (input: Array<number>): number => {
  return sigmoid(dotProduct(weights, input) + bias);
};

const runPerceptron = ({ weights, bias }: Model, input: Array<number>):number => {
  return sigmoid(dotProduct(weights, input) + bias);
};

const generateRandomPerceptron = () => {
  const randomBound = () => 1 - (Math.random() * 2)
  const weights = [ randomBound(), randomBound() ];
  const bias = 0;
  return { weights, bias };
};

const numericalDerivative = (f: number => number) => (x0: number): number => {
  const delta = 0.0001;
  const nd = (f(x0 + delta) - f(x0)) / delta
  return nd;
};


const sampleRandomly = <A>(n: number, array: Array<A>): Array<A> => {
  const ret = [];
  for (let i = 0; i < n; i += 1) {
    const randIndex = Math.floor(Math.random() * array.length);
    ret.push(array[randIndex]);
  }
  return ret;
};

// half of mean squared error
const getCost = (data: Data) => (model: Model): number => {
  const perceptron = makePerceptron(model);
  let acc = 0;
  for (let i = 0; i < data.length; i += 1) {
    const { input, expectedOutput } = data[i];
    const output = perceptron(input);
    acc += Math.pow(output - expectedOutput, 2);
  }
  return acc / (data.length * 2);
};

const getGradientOfCost = (data: Data) => (model: Model) => {
  const perceptron = makePerceptron(model);
  let dc_dw0 = 0;
  let dc_dw1 = 0;
  let dc_db = 0;
  for (let i = 0; i < data.length; i +=1) {
    const { input, expectedOutput } = data[i];
    const { dp_dw0, dp_dw1, dp_db } = gradPerceptron(model, input);
    const diff = perceptron(input) - expectedOutput;
    dc_dw0 += diff * dp_dw0;
    dc_dw1 += diff * dp_dw1;
    dc_db += diff * dp_db;
  }
  dc_dw0 = dc_dw0 / data.length;
  dc_dw1 = dc_dw1 / data.length;
  dc_db = dc_db / data.length;
  return { dc_dw0, dc_dw1, dc_db };
};

const step = (data: Data) => (model: Model): Model => {
  const { dc_dw0, dc_dw1, dc_db } = getGradientOfCost(data)(model);
  const learningRate = 1;
  const nextModel = {
    weights: [
      model.weights[0] - (dc_dw0 * learningRate),
      model.weights[1] - (dc_dw1 * learningRate),
    ],
    bias: model.bias - (dc_db * learningRate),
  };
  return nextModel;
};

const train = (parameters: { steps: number, batchSize: number }) => (data: Data) => (model: Model): Model=> {
  const { steps, batchSize } = parameters;
  let m = model;
  for (let i = 0; i < steps; i += 1 ) {
    const sample = sampleRandomly(batchSize, data);
    const nextModel = step(sample)(m);
    m = nextModel;
  }
  return m;
};

module.exports = {
  runPerceptron,
  generateRandomPerceptron,
  train,
};
