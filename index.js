// @flow

type Continuous<I, O, DO> = {
  run: (I, ...*) => O,
  derivative: (I, ...*) => DO,
};

type Model<M, I, O, DO> = {
  add: (M, M) => M,
  scale: (number, M) => M,
  random: void => M,
  run: (M, I) => O,
  derivative: (M, I) => DO,
} & Continuous<M, O, DO>;

type SimplePerceptron = {
  weights: Array<number>,
  bias: number,
};

const Perceptron: Model<
  SimplePerceptron,
  Array<number>,
  number,
  SimplePerceptron,
> = {
  add: (a, b) => {
    const newWeights = [];
    for (let i = 0; i < a.weights.length; i += 1) {
      newWeights.push(a.weights[i] + b.weights[i]);
    }
    return {
      weights: newWeights,
      bias: a.bias + b.bias,
    };
  },
  scale: (n, model) => {
    return {
      weights: model.weights.map(x => x * n),
      bias: model.bias * n,
    };
  },
  run: (model, input) => {
    const { weights, bias } = model;
    return sigmoid(dotProduct(weights, input) + bias);
  },
  derivative: (model, input) => {
    const { weights, bias } = model;
    const foo = sigmoidPrime(dotProduct(weights, input) + bias);
    return {
      weights: [foo * input[0], foo * input[1]],
      bias: foo,
    };
  },
  random: () => {
    const randomBound = () => 1 - Math.random() * 2;
    const weights = [randomBound(), randomBound()];
    const bias = 0;
    return { weights, bias };
  },
};

type Data = Array<{ input: Array<number>, expectedOutput: number }>;

const setWeight = (i: number, v: number) => (
  model: SimplePerceptron,
): SimplePerceptron => {
  const newWeights = model.weights.slice();
  newWeights[i] = v;
  return {
    weights: newWeights,
    bias: model.bias,
  };
};

const setBias = (v: number) => (model: SimplePerceptron): SimplePerceptron => {
  return {
    weights: model.weights,
    bias: v,
  };
};

const sigmoid = (x: number) => 1 / (1 + Math.exp(-1 * x));

const sigmoidPrime = (x: number) => {
  const s = sigmoid(x);
  return s * (1 - s);
};

const dotProduct = (x: Array<number>, y: Array<number>) => {
  let acc = 0;
  for (let i = 0; i < x.length; i += 1) {
    acc += x[i] * y[i];
  }
  return acc;
};

const sampleRandomly = <A>(n: number, array: Array<A>): Array<A> => {
  const ret = [];
  for (let i = 0; i < n; i += 1) {
    const randIndex = Math.floor(Math.random() * array.length);
    ret.push(array[randIndex]);
  }
  return ret;
};

const halfMeanSquaredError = (
  expected: Array<number>,
  actual: Array<number>,
): number => {
  let acc = 0;
  for (let i = 0; i < expected.length; i += 1) {
    acc += Math.pow(actual[i] - expected[i], 2);
  }
  return acc / (expected.length * 2);
};

// gradient of half MSE wrt. outputs
const halfMeanSquaredErrorGradient = (
  expected: Array<number>,
  actual: Array<number>,
): Array<number> => {
  let grad = [];
  const n = expected.length;
  for (let i = 0; i < n; i += 1) {
    grad.push((actual[i] - expected[i]) / n);
  }
  return grad;
};

// half of mean squared error
const getCost = (data: Data, model: SimplePerceptron): number => {
  const actual = data.map(({ input }) => Perceptron.run(model, input));
  const expected = data.map(({ expectedOutput }) => expectedOutput);
  return halfMeanSquaredError(expected, actual);
};

// Gradient of cost wrt. model params
//
// We multiply the jacobian of the perceptron outputs wrt. the model params
// with the gradient of the loss wrt. the perceptron outputs.
const getGradientOfCost = (
  data: Data,
  model: SimplePerceptron,
): SimplePerceptron => {
  const actual = data.map(({ input }) => Perceptron.run(model, input));
  const expected = data.map(({ expectedOutput }) => expectedOutput);
  const dc_dy = halfMeanSquaredErrorGradient(expected, actual);
  const dp_dtheta = data.map(({ input }) =>
    Perceptron.derivative(model, input),
  );
  let grad = {
    weights: [0, 0],
    bias: 0,
  };
  for (let i = 0; i < data.length; i += 1) {
    grad = Perceptron.add(grad, Perceptron.scale(dc_dy[i], dp_dtheta[i]));
  }
  return grad;
};

const getGradientOfCostNumerically = (
  data: Data,
  model: SimplePerceptron,
): SimplePerceptron => {
  const cost = getCost(data, model);
  const delta = 0.001;
  const dc_dw0 =
    (getCost(data, setWeight(0, model.weights[0] + delta)(model)) - cost) /
    delta;
  const dc_dw1 =
    (getCost(data, setWeight(1, model.weights[1] + delta)(model)) - cost) /
    delta;
  const dc_db =
    (getCost(data, setBias(model.bias + delta)(model)) - cost) / delta;
  return {
    weights: [dc_dw0, dc_dw1],
    bias: dc_db,
  };
};

const step = (data: Data, model: SimplePerceptron): SimplePerceptron => {
  const grad = getGradientOfCost(data, model);
  const learningRate = 1;
  const nextModel = {
    weights: [
      model.weights[0] - grad.weights[0] * learningRate,
      model.weights[1] - grad.weights[1] * learningRate,
    ],
    bias: model.bias - grad.bias * learningRate,
  };
  return nextModel;
};

const train = (parameters: { steps: number, batchSize: number }) => (
  data: Data,
) => (model: SimplePerceptron): SimplePerceptron => {
  const { steps, batchSize } = parameters;
  let m = model;
  for (let i = 0; i < steps; i += 1) {
    const sample = sampleRandomly(batchSize, data);
    const nextModel = step(sample, m);
    m = nextModel;
  }
  return m;
};

module.exports = {
  getGradientOfCost,
  getGradientOfCostNumerically,
  Perceptron,
  train,
};
