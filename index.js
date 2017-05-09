// @flow

type Model = {
  weights: Array<number>,
  bias: number
};

const makePerceptron = ({ weights, bias }: Model) => (input: Array<number>): number => {
  let acc = 0;
  for (let i = 0; i < weights.length; i += 1) {
    acc += weights[i] * input[i];
  }
  return 1 / (1 + Math.exp(-1 * (acc + bias)));
  // if (acc + bias > 0) {
  //   return 1;
  // } else {
  //   return 0;
  // }
};

const generatePerceptronModel = () => {
  const weights = [ Math.random(), Math.random() ];
  const bias = Math.random();
  return { weights, bias };
};


const numericalDerivative = (f: number => number) => (x0: number): number => {
  const delta = 0.0001;
  // console.log(f(x0 + delta), f(x0));
  return (f(x0 + delta) - f(x0))/delta
};

const setW0 = (w0, model): Model => ({
  weights: [
    w0,
    model.weights[1],
  ],
  bias: model.bias,
});

const setW1 = (w1, model): Model => ({
  weights: [
    model.weights[0],
    w1,
  ],
  bias: model.bias,
});

const setB = (b, model): Model => ({
  weights: model.weights,
  bias: b,
});

const mse = (a, b) => ((a - b) * (a - b))/2;

const getCost = (input: Array<number>, expectedOutput: number) => (model: Model): number => {
  const output = makePerceptron(model)(input);
  return mse(output , expectedOutput);
}


const step = (input: Array<number>, expectedOutput: number) => (model: Model): { cost: number, nextModel: Model }=> {
  const _getCost = getCost(input, expectedOutput);
  const cost = _getCost(model); // FIXME: optimize later

  const dw0 = numericalDerivative((w0) => _getCost(setW0(w0, model)))(model.weights[0]);
  const dw1 = numericalDerivative((w1) => _getCost(setW1(w1, model)))(model.weights[1]);
  const db = numericalDerivative((b) => _getCost(setB(b, model)))(model.bias);

  const gamma = 0.001;

  const nextModel = {
    weights: [
      model.weights[0] - dw0 * gamma,
      model.weights[1] - dw1 * gamma,
    ],
    bias: model.bias - db * gamma,
  };

  return {
    cost,
    nextModel,
  };
};

const train = (data: Array<{ input: Array<number>, expectedOutput: number }>) => (model: Model): Model=> {
  let m = model;
  for (let i = 0; i < data.length; i += 1 ) {
    const { input, expectedOutput } = data[i];
    const { cost, nextModel } = step(input, expectedOutput)(m);
    m = nextModel;
  }
  return m
}

module.exports = {
  makePerceptron,
  generatePerceptronModel,
  train,
};
