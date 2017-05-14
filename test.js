// @flow
const {
  runPerceptron,
  generateRandomPerceptron,
  getGradientOfCost,
  getGradientOfCostNumerically,
  train,
} = require("./index.js");
const { expect } = require("chai");

const idealNand = (input: Array<number>) => {
  const perceptron = { weights: [-2, -2], bias: 3 };
  return Math.round(runPerceptron(perceptron, input));
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
    const output = runPerceptron(perceptron, input);
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

describe("runPerceptron", function() {
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
    const model = generateRandomPerceptron();
    const handComputed = getGradientOfCost(data, model);
    const numerical = getGradientOfCostNumerically(data, model);
    const margin = 0.0001;
    expect(handComputed.dc_dw0).to.be.closeTo(numerical.dc_dw0, margin);
    expect(handComputed.dc_dw1).to.be.closeTo(numerical.dc_dw1, margin);
    expect(handComputed.dc_db).to.be.closeTo(numerical.dc_db, margin);
  });
});

describe("learning nand", function() {
  this.timeout(20000);
  describe("train - Stochastic Gradient descent with batch size of 1", function() {
    it("works", function() {
      const perceptron = train({
        steps: 1000000,
        batchSize: 5,
      })(makeData(200000))(generateRandomPerceptron());
      testNand({
        perceptron,
        margin: 1e-5,
      });
    });
  });
});
