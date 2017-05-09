const { makePerceptron, generatePerceptronModel, train } = require('./index.js');
const { expect } = require('chai');

const idealPerceptron = makePerceptron({ weights: [-2, -2], bias: 3 });

describe("makePerceptron", function () {
  describe("and", function () {
    it("works", function () {
      expect(idealPerceptron([1, 0])).to.eql(0.7310585786300049);
      expect(idealPerceptron([0, 1])).to.eql(0.7310585786300049);
      expect(idealPerceptron([0, 0])).to.eql(0.9525741268224331)
      expect(idealPerceptron([1, 1])).to.eql(0.2689414213699951)
    });
  });
});

describe("train", function () {
  describe("nand", function () {
    it("works", function () {
      const data = Array(1000).fill('').map((_, i) => {
        const randomBound = () => 1 - (Math.random() * 2)
        const randomInput = [ randomBound(), randomBound() ];
        return {
          input: randomInput,
          expectedOutput: idealPerceptron(randomInput),
        }
      });
      const model = train(data)(generatePerceptronModel());
      const perceptron = makePerceptron(model);
      expect(perceptron([1, 0])).to.be.closeTo(0.7310585786300049, 0.25);
      expect(perceptron([0, 1])).to.be.closeTo(0.7310585786300049, 0.25);
      expect(perceptron([0, 0])).to.be.closeTo(0.9525741268224331, 0.25);
      expect(perceptron([1, 1])).to.be.closeTo(0.2689414213699951, 0.25);
    });
  });
});
