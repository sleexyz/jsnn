/* export type Continuous<I, O, DO> = {
 *   run: (I, ...*) => O,
 *   derivative: (I, ...*) => DO,
 * };
 */

export type Model<M, I, O, DO> = {
  add: (M, M) => M,
  scale: (number, M) => M,
  initialize: void => M,
  run: (M, I) => O,
  derivative: (M, I) => DO,
};
