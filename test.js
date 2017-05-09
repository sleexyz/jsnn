const { makePerceptron, generatePerceptronModel, train } = require('./index.js');
const { expect } = require('chai');

describe("makePerceptron", function () {
  describe("and", function () {
    it("works", function () {
      const perceptron = makePerceptron({ weights: [1, 1], bias: -1 });
      expect(perceptron([1, 0])).to.eql(0.5);
      expect(perceptron([0, 1])).to.eql(0.5);
      expect(perceptron([0, 0])).to.be.closeTo(0.25, 0.1);
      expect(perceptron([1, 1])).to.be.closeTo(0.75, 0.1);
    });
  });
});   

describe("train", function () {
  describe("nand", function () {
    it("works", function () {
      const data = [
        { input: [1, 0], expectedOutput: 1 },
        { input: [0, 1], expectedOutput: 1 },
        { input: [0, 0], expectedOutput: 1 },
        { input: [1, 1], expectedOutput: 0 },
      ]
      const model = train(data)(generatePerceptronModel());
      const perceptron = makePerceptron(model);
      expect(perceptron([1, 0])).to.be.closeTo(1, 0.5);
      expect(perceptron([0, 1])).to.be.closeTo(1, 0.5);
      expect(perceptron([0, 0])).to.be.closeTo(1, 0.5);
      expect(perceptron([1, 1])).to.be.closeTo(0, 0.5);
    });
  });
});   
