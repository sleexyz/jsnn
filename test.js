// @flow
const {
  makePerceptron,
  generatePerceptronModel,
  train,
} = require('./index.js');
const { expect } = require('chai');

const idealPerceptron = (inputs) => Math.round(makePerceptron({ weights: [-2, -2], bias: 3 })(inputs));

const testNand = ({ margin, perceptron }) => {
  const data = [
    { input: [1, 0], expectedOutput: 1 },
    { input: [0, 1], expectedOutput: 1 },
    { input: [0, 0], expectedOutput: 1 },
    { input: [1, 1], expectedOutput: 0 },
  ];
  for (let i = 0; i < data.length; i += 1) {
    const { input, expectedOutput } = data[i];
    const output = perceptron(input);
    console.log({ input, output });
    expect(output).to.be.closeTo(expectedOutput, margin);
  }
};

const makeData = (length: number) => Array(length).fill('').map((_, i) => {
  const randomBound = () => 1 - (Math.random() * 2)
  const randomInput = [ randomBound(), randomBound() ];
  return {
    input: randomInput,
    expectedOutput: idealPerceptron(randomInput),
  }
});

describe("makePerceptron", function () {
  describe("nand", function () {
    it("works", function () {
      testNand({
        margin: 1e-43,
        perceptron: makePerceptron({ weights: [-200, -200], bias: 300 }),
      });
    });
  });
});

describe("learning nand", function () {
  describe("train - Stochastic Gradient descent with batch size of 1", function () {
    it("works", function () {
      this.timeout(10000);
      const data = makeData(200000);
      const model = train({
        steps: 1000000,
        batchSize: 5,
      })(data)(generatePerceptronModel());
      console.log(model);
      testNand({
        margin: 1e-5,
        perceptron: makePerceptron(model),
      });
    });
  });
});
