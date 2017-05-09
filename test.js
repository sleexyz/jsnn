// @flow
const {
  makePerceptron,
  generatePerceptronModel,
  train0,
  train1,
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

describe("makePerceptron", function () {
  describe("and", function () {
    it("works", function () {
      testNand(idealPerceptron);
    });
  });
});

describe("training methods", function () {
  const data = Array(10000).fill('').map((_, i) => {
    const randomBound = () => 1 - (Math.random() * 2)
    const randomInput = [ randomBound(), randomBound() ];
    return {
      input: randomInput,
      expectedOutput: idealPerceptron(randomInput),
    }
  });

  describe("train0 - SGD with step size 1", function () {
    it("works", function () {
      const model = train0(data)(generatePerceptronModel());
      testNand(makePerceptron(model));
    });
  });

  describe("train1 - GD", function () {
    it("works", function () {
      const model = train1({
        steps: 1000,
      })(data)(generatePerceptronModel());
      testNand(makePerceptron(model));
    });
  });
});
