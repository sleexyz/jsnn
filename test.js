// @flow
const {
  getGradientOfCost,
  getGradientOfCostNumerically,
  Perceptron,
  train,
} = require("./index.js");
const { expect } = require("chai");

const idealNand = (input: Array<number>) => {
  const perceptron = { weights: [-2, -2], bias: 3 };
  return Math.round(Perceptron.run(perceptron, input));
};

const testNand = ({ margin, perceptron }) => {
  const testData = [
    { input: [1, 0], expectedOutput: 1 },
    { input: [0, 1], expectedOutput: 1 },
    { input: [0, 0], expectedOutput: 1 },
    { input: [1, 1], expectedOutput: 0 },
  ];
  for (let i = 0; i < testData.length; i += 1) {
    const { input, expectedOutput } = testData[i];
    const output = Perceptron.run(perceptron, input);
    expect(output).to.be.closeTo(expectedOutput, margin);
  }
};

const makeData = (length: number) =>
  Array(length).fill("").map((_, i) => {
    const randomBound = () => 1 - Math.random() * 2;
    const randomInput = [randomBound(), randomBound()];
    return {
      input: randomInput,
      expectedOutput: idealNand(randomInput),
    };
  });

describe("Perceptron.run", function() {
  describe("nand", function() {
    it("works", function() {
      testNand({
        margin: 1e-43,
        perceptron: { weights: [-200, -200], bias: 300 },
      });
    });
  });
});

describe("getGradientOfCost", function() {
  it("matches the numerical derivative", function() {
    const data = makeData(100);
    const model = Perceptron.initialize();
    const input = data.map(({ input }) => input);
    const expected = data.map(({ expectedOutput }) => expectedOutput);
    const actual = input.map((i) => Perceptron.run(model, i));
    const handComputed = getGradientOfCost(input, actual, expected, model);
    const numerical = getGradientOfCostNumerically(input, actual, expected, model);
    const margin = 0.0001;
    expect(handComputed.weights[0]).to.be.closeTo(numerical.weights[0], margin);
    expect(handComputed.weights[1]).to.be.closeTo(numerical.weights[1], margin);
    expect(handComputed.bias).to.be.closeTo(numerical.bias, margin);
  });
});

describe("learning nand", function() {
  this.timeout(20000);
  describe("train - Stochastic Gradient descent with batch size of 1", function() {
    it("works", function() {
      const perceptron = train({
        steps: 1e5,
        batchSize: 2,
      })(makeData(200000))(Perceptron.initialize());
      testNand({
        perceptron,
        margin: 1e-2,
      });
    });
  });
});
