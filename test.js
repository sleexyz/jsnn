// @flow
const {
  makePerceptron,
  generatePerceptronModel,
  train2,
} = require('./index.js');
const { expect } = require('chai');

const idealPerceptron = makePerceptron({ weights: [-2, -2], bias: 3 });

const testNand = (perceptron) => {
  const margin = 0.1;
  expect(perceptron([1, 0])).to.be.closeTo(0.7310585786300049, margin);
  expect(perceptron([0, 1])).to.be.closeTo(0.7310585786300049, margin);
  expect(perceptron([0, 0])).to.be.closeTo(0.9525741268224331, margin);
  expect(perceptron([1, 1])).to.be.closeTo(0.2689414213699951, margin);
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
      testNand(idealPerceptron);
    });
  });
});

describe("learning nand", function () {
  describe("train2 - Stochastic Gradient descent with batch size of 1", function () {
    it("works", function () {
      const data = makeData(100000);
      const model = train2({
        steps: 10000,
        batchSize: 10,
      })(data)(generatePerceptronModel());
      testNand(makePerceptron(model));
    });
  });
});
