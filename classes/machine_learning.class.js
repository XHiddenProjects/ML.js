//START EXTRA CLASSES
import { SVM } from '../extras/svm.js';
import { NeuralNetwork } from '../extras/neuralNetwork.js';
import { MLMath } from './mlmath.class.js';
import { DBScan } from '../extras/dbscan.js';
// END EXTRA CLASSES

/**
 * Supervised Learning Algorithms
 *
 * @class Supervised 
 */
export const Supervised = class{
    constructor(){

    }
    /**
     * Creates a linear regression
     *
     * @param {Number[]} x X-axis
     * @param {Number[]} y Y-axis
     * @returns {{
     *   slope: number,
     *   intercept: number,
     *   predict: function(number): number,
     *   draw: function(number, number, number): number,
     *   coefficients: { slope: number, intercept: number },
     *   rSquared: number,
     *   residuals: number[],
     *   standardError: number
     * }} 
     * slope - The slope of the line.
     * 
     * intercept - The intercept.
     * 
     * predict - Function to make predictions.
     * 
     * draw - Function to return the data points of the line
     * 
     * coefficients - Object containing slope and intercept.
     * 
     * rSquared - How well the line fits the data.
     * 
     * residuals - Array of residuals for each data point.
     * 
     * standardError - Standard deviation of residuals, indicating prediction accuracy.
     */
    LinearRegression(x, y) {
    if (x.length !== y.length) throw new Error("X and Y do not match");
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }

    const denominator = n * sumX2 - sumX * sumX;
    if (denominator === 0) throw new Error("Denominator is zero, can't compute regression line.");

    const slope = (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;

    // Generate predictions and residuals
    const residuals = [];
    let ssTotal = 0;
    let ssResidual = 0;
    const meanY = sumY / n;

    for (let i = 0; i < n; i++) {
        const yPred = slope * x[i] + intercept;
        residuals.push(y[i] - yPred);
        ssTotal += Math.pow(y[i] - meanY, 2);
        ssResidual += Math.pow(y[i] - yPred, 2);
    }

    // Calculate R-squared
    const rSquared = 1 - ssResidual / ssTotal;

    // Calculate standard error of estimate
    const standardError = Math.sqrt(ssResidual / (n - 2));

    const inverse = 1/slope;

    return {
        slope: slope,
        intercept: intercept,
        predict: (x) => slope * x + intercept,
        draw: (min, max, steps = 1)=>{
            const points = [];
            for (let xVal = min; xVal <= max; xVal += steps) {
                const yVal = slope * xVal + intercept;
                points.push({ x: xVal, y: yVal });
            }
            return points;
        },
        coefficients: { slope: slope, intercept: intercept },
        rSquared: rSquared,
        residuals: residuals,
        standardError: standardError,
        inverse: inverse,
        equation: `${slope.toFixed(4)}*X + ${intercept.toFixed(3)}`,
        meanX: sumX/n,
        meanY: sumY/n,
        correlationCoefficient: Math.sqrt(rSquared) * Math.sign(slope),
        fittedYValues: x.map(xi => slope * xi + intercept),
        residualStdDev: Math.sqrt(residuals.reduce((sum, r) => sum + r * r, 0) / (n - 2))
        };
    }
    /**
     * Creates a logistic regression model
     *
     * @param {Number[][]} X - 2D array of features (each element is an array of feature values for one sample)
     * @param {Number[]} y - Array of labels (0 or 1)
     * @param {{learningRate: number, iterations: number}} [options={}] - Optional settings (learningRate, iterations)
     * @returns {{
     *   weights: Number[],
     *   bias: Number,
     *   predict: function(array): Number,
     *   predictProbability: function(array): Number,
     *   classify: function(array): Number,
     *   drawDecisionBoundary: function(minX, maxX, step): Array
     * }}
     */
    LogisticRegression(X, y, options = {}) {
        const learningRate = options.learningRate || 0.01;
        const iterations = options.iterations || 1000;
        const nSamples = X.length;
        const nFeatures = X[0].length;

        // Initialize weights and bias
        let weights = new Array(nFeatures).fill(0);
        let bias = 0;

        // Sigmoid function
        const sigmoid = (z) => 1 / (1 + Math.exp(-z));

        // Gradient Descent
        for (let iter = 0; iter < iterations; iter++) {
            const dw = new Array(nFeatures).fill(0);
            let db = 0;

            for (let i = 0; i < nSamples; i++) {
                const linearSum = weights.reduce((sum, w, idx) => sum + w * X[i][idx], 0) + bias;
                const yPred = sigmoid(linearSum);
                const error = yPred - y[i];

                // Accumulate gradients
                for (let j = 0; j < nFeatures; j++) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            // Update weights and bias
            for (let j = 0; j < nFeatures; j++) {
                weights[j] -= (learningRate / nSamples) * dw[j];
            }
            bias -= (learningRate / nSamples) * db;
        }

        // Prediction function (probability)
        const predictProbability = (features) => {
            const linearSum = weights.reduce((sum, w, idx) => sum + w * features[idx], 0) + bias;
            return sigmoid(linearSum);
        };

        // Classification (threshold at 0.5)
        const classify = (features, threshold=0.5) => {
            return predictProbability(features) >= threshold ? 1 : 0;
        };

        // For visualization or decision boundary, assume 2D features
        const drawDecisionBoundary = (min, max, step = 0.1) => {
            if (nFeatures !== 2) {
                throw new Error("drawDecisionBoundary is only implemented for 2D features");
            }
            const points = [];
            for (let x1 = min; x1 <= max; x1 += step) {
                // For a 2D line: w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
                const y = -(weights[0] * x1 + bias) / weights[1];
                points.push({ x: x1, y: y });
            }
            return points;
        };

        return {
            weights,
            bias,
            predictProbability,
            classify,
            drawDecisionBoundary
        };
    }
    /**
     * Creates a decision tree classifier
     *
     * @param {Array[]} data - Array of data points, each is an array of feature values with the label as the last element
     * @param {string[]} features - Array of feature names corresponding to each feature index
     * @param {{maxDepth: number, minSamplesSplit: number}} options - Optional settings (maxDepth, minSamplesSplit)
     * @returns {{predict: function(number[]): any, tree: {featureIndex: number, featureName: string, featureValue: number, left: {class: string, type: string}, right:{class: string, type: string}, type: string}, features: any[]}} - The decision tree model with predict method
     */
    DecisionTree(data, features = [], options = {}) {
        const maxDepth = options.maxDepth || 10,
        minSamplesSplit = options.minSamplesSplit || 2,

        // Helper functions
        entropy = (labels) => {
            const counts = {};
            labels.forEach(label => {
                counts[label] = (counts[label] || 0) + 1;
            });
            let ent = 0;
            const total = labels.length;
            for (let label in counts) {
                const p = counts[label] / total;
                ent -= p * Math.log2(p);
            }
            return ent;
        },

        getUniqueValues = (data, index) => {
            const values = new Set();
            data.forEach(row => values.add(row[index]));
            return Array.from(values);
        },

        partition = (data, index, value) => {
            return data.filter(row => row[index] === value);
        },

        // Recursive function to build the tree
        buildTree = (data, depth = 0) => {
            const labels = data.map(row => row[row.length - 1]),
            numSamples = data.length,
            numLabels = new Set(labels).size;

            // Stopping conditions
            if (depth >= maxDepth || numSamples < minSamplesSplit || numLabels === 1) {
                // Return leaf node with majority class
                const counts = {};
                labels.forEach(label => {
                    counts[label] = (counts[label] || 0) + 1;
                });
                const labelsArray = Object.keys(counts),
                majorityLabel = labelsArray.length > 0 
                    ? labelsArray.reduce((a, b) => counts[a] > counts[b] ? a : b)
                    : null; // fallback if labels are empty
                return { type: 'leaf', class: majorityLabel };
            }

            // Find the best split
            const nFeatures = data[0].length - 1;
            let bestFeatureIdx = -1,
            bestGain = -Infinity,
            bestValue = null;

            const baseEntropy = entropy(labels);

            for (let featureIdx = 0; featureIdx < nFeatures; featureIdx++) {
                const values = getUniqueValues(data, featureIdx);
                for (let val of values) {
                    const subset1 = partition(data, featureIdx, val),
                    subset2 = data.filter(row => row[featureIdx] !== val),
                    weight1 = subset1.length / data.length,
                    weight2 = subset2.length / data.length,
                    newEntropy = weight1 * entropy(subset1.map(row => row[row.length - 1]))
                                    + weight2 * entropy(subset2.map(row => row[row.length - 1])),
                    infoGain = baseEntropy - newEntropy;
                    if (infoGain > bestGain) {
                        bestGain = infoGain;
                        bestFeatureIdx = featureIdx;
                        bestValue = val;
                    }
                }
            }

            if (bestFeatureIdx === -1) {
                // No good split found, return majority class
                const counts = {};
                labels.forEach(label => {
                    counts[label] = (counts[label] || 0) + 1;
                });
                const majorityLabel = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
                return { type: 'leaf', class: majorityLabel };
            }

            // Build subtrees
            const leftSubset = partition(data, bestFeatureIdx, bestValue),
            rightSubset = data.filter(row => row[bestFeatureIdx] !== bestValue);

            return {
                type: 'node',
                featureIndex: bestFeatureIdx,
                featureName: features[bestFeatureIdx] || `Feature ${bestFeatureIdx}`,
                featureValue: bestValue,
                left: buildTree(leftSubset, depth + 1),
                right: buildTree(rightSubset, depth + 1)
            };
        };

        // Prediction function
        const predict = (sample, node) => {
            if (node.type === 'leaf') {
                return node.class;
            }
            // Determine which branch to follow
            if (sample[node.featureIndex] === node.featureValue) {
                return predict(sample, node.left);
            } else {
                return predict(sample, node.right);
            }
        },

        tree = buildTree(data);

        return {
            predict: (sample) => predict(sample, tree),
            tree: tree,
            features: features
        };
    }
    /**
     * Creates a Random Forest classifier
     *
     * @param {Array[]} data - Array of data points, each with features and label as the last element
     * @param {string[]} features - Array of feature names
     * @param {Object} options - Optional settings (nEstimators, maxDepth, minSamplesSplit, maxFeatures)
     * @returns {{
     *   predict: function(sample): any,
     *   trees: Array,
     *   nEstimators: number,
     *   featureSubsetSize: number
     * }}
     */
    RandomForests(data, features = [], options = {}) {
        const nEstimators = options.nEstimators || 10, // Number of trees
        maxDepth = options.maxDepth || 10,
        minSamplesSplit = options.minSamplesSplit || 2,
        maxFeatures = options.maxFeatures || Math.floor(Math.sqrt(features.length)) || 1,

        trees = [],

        // Helper function to get a bootstrap sample
        bootstrapSample = (data) => {
            const sample = [];
            for (let i = 0; i < data.length; i++) {
                const idx = Math.floor(Math.random() * data.length);
                sample.push(data[idx]);
            }
            return sample;
        },

        // Helper to select a random subset of features
        getFeatureIndices = () => {
            const indices = [],
            totalFeatures = features.length;
            for (let i = 0; i < maxFeatures; i++) {
                const idx = Math.floor(Math.random() * totalFeatures);
                if (!indices.includes(idx)) {
                    indices.push(idx);
                }
            }
            return indices;
        };

        // Train multiple trees
        for (let i = 0; i < nEstimators; i++) {
            const sampleData = bootstrapSample(data),
            // For each tree, select a subset of features
            featureIndices = getFeatureIndices(),

            // Extract relevant features for training
            trainData = sampleData.map(row => {
                const featuresSubset = featureIndices.map(idx => row[idx]);
                return [...featuresSubset, row[row.length - 1]]; // Append label
            }),

            // Train a decision tree on this subset
            tree = this.DecisionTree(trainData, featureIndices.map(idx => features[idx]), {
                maxDepth: maxDepth,
                minSamplesSplit: minSamplesSplit
            });

            // Store the tree along with its feature subset
            trees.push({ tree: tree, featureIndices: featureIndices });
        }

        // Prediction by majority voting
        const predict = (sample) => {
            const votes = {};
            for (const { tree, featureIndices } of trees) {
                const sampleFeatures = featureIndices.map(idx => sample[idx]),
                prediction = tree.predict(sampleFeatures);
                votes[prediction] = (votes[prediction] || 0) + 1;
            }
            // Return the class with the most votes
            return Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);
        };

        return {
            predict: predict,
            trees: trees,
            nEstimators: nEstimators,
            featureSubsetSize: maxFeatures
        };
    }
    
    /**
     * SVM (Support Vector Machines)
     *
     * @param {Array<Array<number>>} features - Training features (array of feature vectors)
     * @param {Array<number>} labels - Training labels
     * @param {{cost: Number, 
     * tol: Number, 
     * maxPasses: Number, 
     * maxIterations: Number, 
     * kernel: 'linear'|'poly'|'rbf'|'sigmoid', 
     * alphaTol:Number, 
     * random: Math.random, 
     * kernelOptions: Object,
     * positiveClass: number}} [options={}] - Configuration options for the SVM
     * @returns {SVM}
     */
    SVM(features, labels, options = {}) {
        const svm = new SVM(options);
        svm.train(features,labels);
        return svm;
    }
    /**
     * k-Nearest Neighbors (kNN) classifier function.
     * 
     * This function takes a set of training data points, a test point,
     * and a value of 'k' to determine the most common label among
     * the 'k' closest neighbors using Euclidean distance.
     * 
     * @param {Array<{features: [], label:string}>} trainingData - Array of objects with 'features' (array) and 'label'
     * @param {number[]} testPoint - Object with 'features' (array) to classify
     * @param {number} k - Number of neighbors to consider
     * @returns {{predictedLabel: String, distance: Number, neighborCount: Number}} Predicted label, distance, and neighbor count
     */
    KNN(trainingData, testPoint, k) {
        // Array to hold distances and corresponding labels
        const distances = [];

        // Loop through each training data point
        for (let i = 0; i < trainingData.length; i++) {
            const dataPoint = trainingData[i];
            const distance = MLMath.distance(dataPoint.features, testPoint,'euclidean');
            distances.push({ distance: distance, label: dataPoint.label });
        }

        // Sort the array by distance in ascending order
        distances.sort((a, b) => a.distance - b.distance);
        
        // Select the top k nearest neighbors
        const neighbors = distances.slice(0, k);

        // Count the frequency of each label among neighbors
        const labelCounts = {};
        for (let neighbor of neighbors) {
            labelCounts[neighbor.label] = (labelCounts[neighbor.label] || 0) + 1;
        }

        // Determine the label with the highest count
        let maxCount = -1;
        let predictedLabel = null;
        for (let label in labelCounts) {
            if (labelCounts[label] > maxCount) {
                maxCount = labelCounts[label];
                predictedLabel = label;
            }
        }

        // Find the distance of the predicted label among neighbors
        // Usually, this is the distance of the closest neighbor with the predicted label
        const predictedNeighbors = neighbors.filter(n => n.label === predictedLabel);
        const distanceToPredictedLabel = predictedNeighbors.length > 0 ? predictedNeighbors[0].distance : null;

        return {
            predictedLabel: predictedLabel,
            distance: distanceToPredictedLabel,
            neighborCount: maxCount
        };
    }
    
    /**
     * Initializes and trains a neural network with the provided dataset, features, and options.
     *
     * @param {Array<Object>} data - Dataset array of objects.
     * @param {Array<string>} features - List of feature names to use.
     * @param {Object} [options={}] - Configuration options.
     * @param {number} [options.hiddenLayers=1] - Number of hidden layers.
     * @param {number} [options.epoch=1000] - Number of training epochs.
     * @param {number} [options.learningRate=0.1] - Learning rate.
     * @param {'ReLU'|'Tanh'|'Sigmoid'|'Linear'} [options.activation='Sigmoid'] - Activation function.
     * @param {'none'|'L1'|'L2'} [options.regularization='none'] - Regularization type.
     * @param {number} [options.regularizationRate=0.01] - Regularization rate.
     * @param {number} [options.noise=0] - Noise level to add to inputs.
     * @param {number} [options.batch=15] - Batch size for training.
     * @param {number} [options.ratioTrainTest=0.8] - Train-test split ratio.
     * @param {'classification'|'regression'} [options.type='regression'] - Output type.
     * @param {boolean} [options.debug=false] - Debug in console
     * @param {boolean} [options.ones=false] - Set biases with 1's intend of 0's
     * @returns {NeuralNetwork} - Neural Network trained
     */
    NeuralNetwork(data, features, options) {
        const nn = new NeuralNetwork(data,features,options);
        nn.fit()
        return nn;
    }
    
    /**
     * Gradient Boosting Machine (GBM) implementation for regression tasks.
     *
     * @param {Array} data - Training data where each row is an array of feature values with the target as the last element.
     * @param {{}} [features=[]] - Feature names corresponding to each feature index.
     * @param {{nEstimators: number, learningRate: number, maxDepth:number, min_sample_split:number, min_samples_leaf:number, subsample:number, max_features:number}} [options={}] - Configuration options.
     * @returns {{ predict: (sample: any) => number; predictAll(samples: any): any; initialPrediction: number; stumps: {}; nEstimators: any; learningRate: any; features: {}; }} 
     */
    GBM(data, features = [], options = {}) {
        const nEstimators = options.nEstimators || 100;
        const learningRate = options.learningRate || 0.1;
        const maxDepth = options.maxDepth || 1; // currently only stumps (depth=1) are supported
        const minSamplesSplit = options.min_sample_split || options['minSamplesSplit'] || 2;
        const minSamplesLeaf = options.min_samples_leaf || options['minSamplesLeaf'] || 1;
        const subsample = typeof options.subsample === 'number' ? Math.max(0, Math.min(1, options.subsample)) : 1.0;

        if (!Array.isArray(data) || data.length === 0) {
            throw new Error("Data must be a non-empty array of rows where the last element is the target.");
        }

        // Prepare X and y (assume each row is [...features, target])
        const X = data.map(row => row.slice(0, -1));
        const y = data.map(row => row[row.length - 1]);

        const nSamples = X.length;
        const nFeatures = X[0] ? X[0].length : 0;

        // Determine max_features (number of features to consider for each stump)
        const maxFeatures = options.max_features || options['maxFeatures'] || Math.max(1, Math.floor(Math.sqrt(nFeatures)) || nFeatures);

        // If user requested deeper trees, reject because current implementation supports stumps only.
        if (maxDepth !== 1) {
            throw new Error("Only stumps (maxDepth = 1) are supported in this GBM implementation.");
        }

        // Initial prediction is the mean of targets
        const initialPrediction = y.reduce((a, b) => a + b, 0) / nSamples;
        let yPred = new Array(nSamples).fill(initialPrediction);

        const stumps = [];

        // Helper: sample row indices for subsample (without replacement)
        const getSampleIndices = () => {
            const sampleSize = Math.max(1, Math.floor(nSamples * subsample));
            const indices = Array.from({ length: nSamples }, (_, i) => i);
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }
            return indices.slice(0, sampleSize);
        };

        // Helper: choose feature indices for this stump (without replacement)
        const getFeatureIndices = () => {
            const all = Array.from({ length: nFeatures }, (_, i) => i);
            for (let i = all.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [all[i], all[j]] = [all[j], all[i]];
            }
            return all.slice(0, Math.min(maxFeatures, all.length));
        };

        // Fit a regression stump (one-level tree) to residuals
        function fitStump(xLocal, residuals, featureIndices) {
            let best = null;

            for (const j of featureIndices) {
                // Pair feature values with residuals
                const pairs = xLocal.map((xi, idx) => ({ x: xi[j], r: residuals[idx] }));

                // Unique sorted feature values
                const values = Array.from(new Set(pairs.map(p => p.x))).sort((a, b) => a - b);

                if (values.length === 0) continue;
                if (values.length === 1) {
                    // Single-value feature: predict mean residual
                    const meanVal = pairs.reduce((s, p) => s + p.r, 0) / pairs.length;
                    const sse = pairs.reduce((s, p) => s + Math.pow(p.r - meanVal, 2), 0);
                    if (!best || sse < best.sse) {
                        best = {
                            featureIndex: j,
                            threshold: values[0],
                            leftValue: meanVal,
                            rightValue: meanVal,
                            sse: sse
                        };
                    }
                    continue;
                }

                // Consider thresholds between adjacent unique values
                for (let t = 0; t < values.length - 1; t++) {
                    const threshold = (values[t] + values[t + 1]) / 2;
                    let leftSum = 0, leftCount = 0, rightSum = 0, rightCount = 0;

                    for (const p of pairs) {
                        if (p.x <= threshold) {
                            leftSum += p.r;
                            leftCount++;
                        } else {
                            rightSum += p.r;
                            rightCount++;
                        }
                    }

                    // Respect minSamplesSplit and minSamplesLeaf constraints
                    const totalCount = leftCount + rightCount;
                    if (totalCount < minSamplesSplit) continue;
                    if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) continue;

                    const leftMean = leftCount ? leftSum / leftCount : 0;
                    const rightMean = rightCount ? rightSum / rightCount : 0;

                    let sse = 0;
                    for (const p of pairs) {
                        const pred = p.x <= threshold ? leftMean : rightMean;
                        sse += Math.pow(p.r - pred, 2);
                    }

                    if (!best || sse < best.sse) {
                        best = {
                            featureIndex: j,
                            threshold: threshold,
                            leftValue: leftMean,
                            rightValue: rightMean,
                            sse: sse
                        };
                    }
                }
            }

            if (!best) return null;

            return {
                featureIndex: best.featureIndex,
                threshold: best.threshold,
                leftValue: best.leftValue,
                rightValue: best.rightValue,
                predict(sample) {
                    return sample[this.featureIndex] <= this.threshold ? this.leftValue : this.rightValue;
                }
            };
        }

        // Iteratively fit stumps on residuals
        for (let m = 0; m < nEstimators; m++) {
            // Subsample rows
            const sampleIdx = getSampleIndices();
            const xSub = sampleIdx.map(i => X[i]);
            const residualsSub = sampleIdx.map(i => y[i] - yPred[i]);

            // Choose feature subset for this stump
            const featureIndices = getFeatureIndices();
            const stump = fitStump(xSub, residualsSub, featureIndices);
            if (!stump) break;

            // Update predictions with the new stump scaled by learning rate
            for (let i = 0; i < nSamples; i++) {
                yPred[i] += learningRate * stump.predict(X[i]);
            }

            stumps.push(stump);
        }

        // Prediction function for a single sample (array of feature values)
        const predict = (sample) => {
            let pred = initialPrediction;
            for (const s of stumps) {
                pred += learningRate * s.predict(sample);
            }
            return pred;
        };

        return {
            predict,
            predictAll(samples) {
                if (!Array.isArray(samples)) return [];
                return samples.map(s => predict(s));
            },
            initialPrediction,
            stumps,
            nEstimators: stumps.length,
            learningRate,
            features
        };
    }
    /**
     * AdaBoost classifier using decision stumps as weak learners.
     *
     * @param {Array[]} data - Array of rows where each row is [...features, label] and label is 0/1 or -1/1.
     * @param {{nEstimators:number}} [options={}] - Options object; supports nEstimators (default 50).
     * @returns {{
     *   predict: function(Array<number>): number,
     *   predictScore: function(Array<number>): number,
     *   predictProbability: function(Array<number>): number,
     *   stumps: Array,
     *   alphas: Array,
     *   nEstimators: number
     * }} AdaBoost model with prediction helpers.
     */
    AdaBoost(data, options = {}) {
        const requestedNEstimators = options.nEstimators || 50;
        const nEstimators = requestedNEstimators;
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error("Data must be a non-empty array of rows where the last element is the label.");
        }

        const X = data.map(r => r.slice(0, -1));
        const yRaw = data.map(r => r[r.length - 1]);
        const nSamples = X.length;
        const nFeatures = X[0] ? X[0].length : 0;

        if (nFeatures === 0) {
            throw new Error("Data rows must contain at least one feature plus a label.");
        }

        // Map labels to -1 / +1
        const y = yRaw.map(v => {
            if (v === -1) return -1;
            if (v === 0) return -1;
            if (v === 1) return 1;
            return v > 0 ? 1 : -1;
        });

        // Initialize sample weights
        let w = new Array(nSamples).fill(1 / nSamples);

        const stumps = [];
        const alphas = [];

        const uniqueSorted = (arr) => Array.from(new Set(arr
            .map(v => Number(v))
            .filter(v => isFinite(v))
        )).sort((a, b) => a - b);

        const EPS = 1e-12;

        for (let m = 0; m < nEstimators; m++) {
            let best = null;

            // Find best decision stump across all features and thresholds
            for (let j = 0; j < nFeatures; j++) {
                const vals = uniqueSorted(X.map(row => row[j]));
                if (vals.length === 0) continue;

                const thresholds = [];
                if (vals.length === 1) {
                    // single unique value: try a threshold slightly below and above to allow splits
                    thresholds.push(vals[0] - 1e-6);
                    thresholds.push(vals[0]);
                    thresholds.push(vals[0] + 1e-6);
                } else {
                    for (let t = 0; t < vals.length - 1; t++) {
                        thresholds.push((vals[t] + vals[t + 1]) / 2);
                    }
                    // Add extreme thresholds to allow all-one-side splits
                    thresholds.push(vals[0] - 1e-6);
                    thresholds.push(vals[vals.length - 1] + 1e-6);
                }

                for (const threshold of thresholds) {
                    // polarity = 1 => predict +1 when x <= threshold, else -1
                    // polarity = -1 => flip
                    for (const polarity of [1, -1]) {
                        let error = 0;
                        for (let i = 0; i < nSamples; i++) {
                            const xi = Number(X[i][j]);
                            const pred = (xi <= threshold ? 1 : -1) * polarity;
                            if (pred !== y[i]) error += w[i];
                        }
                        if (!best || error < best.error) {
                            best = { featureIndex: j, threshold: Number(threshold), polarity, error };
                        }
                    }
                }
            }

            if (!best) break;

            // Clamp error to avoid division by zero or log of zero
            let err = Math.max(EPS, Math.min(1 - EPS, best.error));

            // If error is too close to 0 => treat as near-perfect but bounded
            if (best.error < EPS) {
                const alpha = 0.5 * Math.log((1 - EPS) / EPS);
                stumps.push(best);
                alphas.push(alpha);
                break;
            }

            // If stump is not better than random, stop adding learners
            if (err >= 0.5) {
                break;
            }

            const alpha = 0.5 * Math.log((1 - err) / err);

            // Update weights with numerical stability
            for (let i = 0; i < nSamples; i++) {
                const xi = Number(X[i][best.featureIndex]);
                const pred = (xi <= best.threshold ? 1 : -1) * best.polarity;
                // multiply by exp(-alpha * y * pred)
                w[i] = w[i] * Math.exp(-alpha * y[i] * pred);
                if (!isFinite(w[i]) || w[i] <= 0) w[i] = EPS;
            }
            // Normalize weights
            const sumW = w.reduce((a, b) => a + b, 0) || EPS;
            for (let i = 0; i < nSamples; i++) w[i] /= sumW;

            stumps.push(best);
            alphas.push(alpha);
        }

        // Score is the aggregated weighted sum of stump predictions
        const score = (sample) => {
            // Accept either a features array (e.g. [x1,x2]) or a full data row [x1,x2,label]
            if (!Array.isArray(sample)) throw new Error("Sample must be an array of feature values (optionally with label as last element).");
            const sFeatures = sample.length === nFeatures + 1 ? sample.slice(0, nFeatures) : (sample.length >= nFeatures ? sample.slice(0, nFeatures) : null);
            if (!Array.isArray(sFeatures) || sFeatures.length !== nFeatures) {
                throw new Error("Sample length does not match number of features.");
            }
            let s = 0;
            for (let k = 0; k < stumps.length; k++) {
                const st = stumps[k];
                const val = Number(sFeatures[st.featureIndex]);
                const pred = (val <= st.threshold ? 1 : -1) * st.polarity;
                s += (alphas[k] || 0) * pred;
            }
            return s;
        };

        // Predict label (0/1)
        const predict = (sample) => {
            const s = score(sample);
            return s >= 0 ? 1 : 0;
        };

        // Predict probability using a logistic mapping of the margin (numerically stable)
        const predictProbability = (sample) => {
            const s = score(sample);
            // use clamped s to avoid overflow in exp
            const S = Math.max(-100, Math.min(100, s));
            return 1 / (1 + Math.exp(-2 * S));
        };

        return {
            predict,
            predictScore: score,
            predictProbability,
            stumps,
            alphas,
            nEstimators: stumps.length,
            requestedNEstimators
        };
    }
}

/**
 * Unsupervised Learning Algorithms
 *
 * @class Unsupervised 
 */
export const Unsupervised = class {
    constructor(){

    }
    
    /**
     * K-Means Clustering algorithm.
     *
     * @param {Array.<Array.<number>>|Array.<*>} data - Array of data points, each is an array of feature values (optionally with a label as first element)
     * @param {number} k - Number of clusters
     * @param {Object} [options={}] - Optional settings
     * @param {number} [options.maxIterations] - Maximum number of iterations
     * @param {number} [options.tolerance] - Convergence tolerance
     * @param {boolean} [options.allowLabels] - Whether the first column contains labels
     * @returns {{centroids: Array.<Array.<number>>, assignments: Array.<number>, iterations: number, labels: Array|undefined}} 
     */
    KMeansClustering(data, k, options = {}) {
        const maxIterations = options.maxIterations || 100;
        const tolerance = options.tolerance || 1e-4;

        if (!Array.isArray(data) || data.length === 0) {
            throw new Error("Data must be a non-empty array of points.");
        }

        const nSamples = data.length;

        // Determine if data rows include a label as the first element.
        // If options.allowLabels is true, or all rows have a non-number first element,
        // treat the first column as labels and ignore it for clustering.
        const maybeHasLabel = !!options.allowLabels || data.every(row => typeof row[0] === 'string');
        const X = maybeHasLabel ? data.map(row => row.slice(1)) : data.map(row => row.slice());
        const labels = maybeHasLabel ? data.map(row => row[0]) : null;

        if (X.length === 0 || !Array.isArray(X[0])) {
            throw new Error("Data rows must be arrays of feature values.");
        }

        const nFeatures = X[0].length;

        // Cap k to number of samples to avoid infinite loops / invalid initialization
        k = Math.max(1, Math.min(k, nSamples));

        // Initialize centroids by sampling k distinct indices (or reuse if less available)
        const indices = Array.from({ length: nSamples }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        let centroids = indices.slice(0, k).map(i => [...X[i]]);

        let assignments = new Array(nSamples).fill(-1);
        let iterations = 0;
        let converged = false;

        while (!converged && iterations < maxIterations) {
            converged = true;

            // Assignment step
            for (let i = 0; i < nSamples; i++) {
                let minDist = Infinity;
                let bestCluster = -1;
                for (let j = 0; j < k; j++) {
                    const dist = MLMath.distance(X[i], centroids[j],'euclidean');
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }
                if (assignments[i] !== bestCluster) {
                    converged = false;
                    assignments[i] = bestCluster;
                }
            }

            // Update step
            const newCentroids = Array.from({ length: k }, () => new Array(nFeatures).fill(0));
            const counts = new Array(k).fill(0);

            for (let i = 0; i < nSamples; i++) {
                const cluster = assignments[i];
                if (cluster < 0 || cluster >= k) continue;
                counts[cluster]++;
                for (let f = 0; f < nFeatures; f++) {
                    newCentroids[cluster][f] += X[i][f];
                }
            }

            for (let j = 0; j < k; j++) {
                if (counts[j] === 0) {
                    // Reinitialize empty centroid to a random data point to avoid empty clusters
                    newCentroids[j] = [...X[Math.floor(Math.random() * nSamples)]];
                    continue;
                }
                for (let f = 0; f < nFeatures; f++) {
                    newCentroids[j][f] /= counts[j];
                }
            }

            // Check for convergence
            let maxShift = 0;
            for (let j = 0; j < k; j++) {
                const shift = MLMath.distance(centroids[j], newCentroids[j],'euclidean');
                if (shift > maxShift) {
                    maxShift = shift;
                }
            }
            centroids = newCentroids;
            if (maxShift > tolerance) {
                converged = false;
            }
            iterations++;
        }

        return {
            centroids,
            assignments,
            iterations,
            labels: labels || undefined
        };
    }
    /**
     * Agglomerative Hierarchical Clustering (basic implementation).
     *
     * @param {Array.<Array.<number>>|Array.<*>} data - Array of data points (optionally with a label as first element if options.allowLabels=true)
     * @param {Object} [options={}] - Options object.
     * @param {'single'|'complete'|'average'} [options.linkage='single'] - Linkage criterion.
     * @param {'euclidean'|'manhattan'} [options.metric='euclidean'] - Distance metric.
     * @param {number} [options.nClusters=2] - Desired number of clusters.
     * @param {boolean} [options.allowLabels=false] - If true, treat first column as label and ignore in clustering.
     * @returns {{
     *   clusters: Array.<Array.<{index:number, point:Array<number>, label: any}>>,
     *   clusterIndices: Array.<Array.<number>>,
     *   linkage: string,
     *   metric: string,
     *   nClusters: number
     * }} Result object containing clusters and metadata.
     */
    HierarchicalClustering(data, options = {}) {
        const linkage = options.linkage || 'single';
        const metric = options.metric || 'euclidean';
        const requestedClusters = typeof options.nClusters === 'number' ? Math.max(1, Math.floor(options.nClusters)) : 2;
        const allowLabels = !!options.allowLabels;

        if (!Array.isArray(data) || data.length === 0) {
            throw new Error("Data must be a non-empty array of points.");
        }

        // Extract labels if requested
        const labels = allowLabels ? data.map(row => row[0]) : null;
        const X = allowLabels ? data.map(row => row.slice(1)) : data.map(row => row.slice());

        if (!Array.isArray(X[0]) || X[0].length === 0) {
            throw new Error("Data rows must be arrays of feature values.");
        }

        const n = X.length;


        // Compute distance between two clusters according to linkage
        function clusterDistance(ci, cj) {
            let values = [];
            for (const i of ci) {
                for (const j of cj) {
                    values.push(MLMath.distance(X[i], X[j], metric));
                }
            }
            if (values.length === 0) return Infinity;
            if (linkage === 'complete') return Math.max(...values);
            if (linkage === 'average') return values.reduce((s, v) => s + v, 0) / values.length;
            // default single linkage
            return Math.min(...values);
        }

        // Initialize each point as its own cluster (store indices)
        let clusters = [];
        for (let i = 0; i < n; i++) clusters.push([i]);

        // Agglomerative merging until we reach requestedClusters
        while (clusters.length > requestedClusters) {
            let bestI = -1, bestJ = -1, bestDist = Infinity;
            for (let i = 0; i < clusters.length; i++) {
                for (let j = i + 1; j < clusters.length; j++) {
                    const d = clusterDistance(clusters[i], clusters[j]);
                    if (d < bestDist) {
                        bestDist = d;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }
            if (bestI === -1 || bestJ === -1) break;
            // Merge clusters bestI and bestJ (ensure remove higher index first)
            const merged = clusters[bestI].concat(clusters[bestJ]);
            // Remove the two clusters and insert merged
            if (bestI > bestJ) {
                clusters.splice(bestI, 1);
                clusters.splice(bestJ, 1);
            } else {
                clusters.splice(bestJ, 1);
                clusters.splice(bestI, 1);
            }
            clusters.push(merged);
        }

        // Build descriptive cluster objects
        const clusterObjects = clusters.map(clusterIdxs => {
            return clusterIdxs.map(idx => ({
                index: idx,
                point: X[idx],
                label: labels ? labels[idx] : undefined
            }));
        });

        return {
            clusters: clusterObjects,
            clusterIndices: clusters,
            linkage,
            metric,
            nClusters: clusters.length
        };
    }
    /**
     * Principal Component Analysis (PCA)
     *
     * @param {Array.<Array.<number>>} data - 2D array with shape [nSamples][nFeatures]
     * @param {Object} [options] - Optional settings
     * @param {number} [options.nComponents] - Number of principal components to keep (default: all)
     * @param {number} [options.maxIter] - Max iterations for power method (default: 1000)
     * @param {number} [options.tol] - Tolerance for power method convergence (default: 1e-9)
     * @returns {{
     *   components: Array.<Array.<number>>,      // principal axes (nComponents x nFeatures)
     *   explainedVariance: Array.<number>,       // variance explained by each component
     *   mean: Array.<number>,                    // per-feature mean used for centering
     *   transform: function(Array.<Array.<number>>): Array.<Array.<number>>,
     *   inverseTransform: function(Array.<Array.<number>>): Array.<Array.<number>>,
     *   project: function(Array.<number>): Array.<number>,
     *   explainedVarianceRatio: Array.<number>
     * }}
     */
    PCA(data, options = {}) {
        const nComponents = options.nComponents || null;
        const maxIter = options.maxIter || 1000;
        const tol = typeof options.tol === 'number' ? options.tol : 1e-9;

        if (!Array.isArray(data) || data.length === 0 || !Array.isArray(data[0])) {
            throw new Error("PCA expects a non-empty 2D array `data` of shape [nSamples][nFeatures].");
        }

        const nSamples = data.length;
        const nFeatures = data[0].length;
        const k = nComponents ? Math.min(Math.max(1, Math.floor(nComponents)), nFeatures) : nFeatures;

        // Helpers
        const meanVec = (X) => {
            const m = new Array(nFeatures).fill(0);
            for (let i = 0; i < X.length; i++) {
                const row = X[i];
                for (let j = 0; j < nFeatures; j++) m[j] += row[j];
            }
            for (let j = 0; j < nFeatures; j++) m[j] /= X.length;
            return m;
        };

        const centerData = (X, mean) => X.map(row => row.map((v, i) => v - mean[i]));

        const transpose = (M) => {
            const r = M.length, c = M[0].length;
            const T = Array.from({ length: c }, () => new Array(r));
            for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = M[i][j];
            return T;
        };

        const matMul = (A, B) => {
            const aR = A.length, aC = A[0].length, bC = B[0].length;
            const C = Array.from({ length: aR }, () => new Array(bC).fill(0));
            for (let i = 0; i < aR; i++) {
                for (let k2 = 0; k2 < aC; k2++) {
                    const aik = A[i][k2];
                    for (let j = 0; j < bC; j++) {
                        C[i][j] += aik * B[k2][j];
                    }
                }
            }
            return C;
        };

        const dot = (a, b) => {
            let s = 0;
            for (let i = 0; i < a.length; i++) s += a[i] * b[i];
            return s;
        };

        const vecNorm = (v) => Math.sqrt(dot(v, v)) || 1e-16;

        const matVecMul = (M, v) => {
            const r = M.length;
            const out = new Array(r).fill(0);
            for (let i = 0; i < r; i++) {
                let s = 0;
                const row = M[i];
                for (let j = 0; j < row.length; j++) s += row[j] * v[j];
                out[i] = s;
            }
            return out;
        };

        // Compute mean and center the data
        const mean = meanVec(data);
        const Xc = centerData(data, mean);

        // Compute covariance matrix (features x features)
        // cov = (Xc^T * Xc) / (nSamples - 1)
        const Xt = transpose(Xc);
        const cov = matMul(Xt, Xc).map(row => row.map(v => v / Math.max(1, nSamples - 1)));

        // Power iteration with deflation to get top-k eigenpairs
        const components = []; // will store eigenvectors (each length nFeatures)
        const eigenvalues = [];

        // Make a working copy of covariance for deflation
        const covWork = cov.map(r => r.slice());

        for (let comp = 0; comp < k; comp++) {
            // Initialize random vector
            let v = new Array(nFeatures).fill(0).map(() => Math.random() - 0.5);
            let lambda = 0;
            let iter = 0;
            let converged = false;

            while (iter < maxIter && !converged) {
                // Multiply by matrix
                const w = matVecMul(covWork, v);
                const norm = vecNorm(w);
                if (norm === 0) break;
                // Normalize
                for (let i = 0; i < v.length; i++) v[i] = w[i] / norm;
                // Rayleigh quotient for eigenvalue estimate
                const w2 = matVecMul(covWork, v);
                const lambdaNew = dot(v, w2);
                if (Math.abs(lambdaNew - lambda) < tol) converged = true;
                lambda = lambdaNew;
                iter++;
            }

            // Normalize final eigenvector
            const vNorm = vecNorm(v);
            for (let i = 0; i < v.length; i++) v[i] /= vNorm;

            // Store
            components.push(v.slice());
            eigenvalues.push(lambda);

            // Deflate: covWork = covWork - lambda * (v outer v)
            for (let i = 0; i < nFeatures; i++) {
                for (let j = 0; j < nFeatures; j++) {
                    covWork[i][j] -= lambda * v[i] * v[j];
                }
            }
        }

        const totalVariance = eigenvalues.reduce((s, a) => s + a, 0) || 1e-16;
        const explainedVarianceRatio = eigenvalues.map(ev => ev / totalVariance);
        const explainedVariance = eigenvalues.slice();

        // Transform function: project centered samples to component space (nSamples x k)
        const transform = (samples) => {
            if (!Array.isArray(samples)) throw new Error("transform expects an array of samples");
            const out = samples.map(row => {
                const centered = row.map((v, i) => v - mean[i]);
                return components.map(c => dot(centered, c));
            });
            return out;
        };

        // inverseTransform: reconstruct approximate original features from projected data
        const inverseTransform = (projected) => {
            if (!Array.isArray(projected)) throw new Error("inverseTransform expects an array of projected rows");
            return projected.map(coords => {
                const recon = new Array(nFeatures).fill(0);
                for (let i = 0; i < coords.length && i < components.length; i++) {
                    const coeff = coords[i];
                    const compVec = components[i];
                    for (let j = 0; j < nFeatures; j++) recon[j] += coeff * compVec[j];
                }
                // add mean back
                for (let j = 0; j < nFeatures; j++) recon[j] += mean[j];
                return recon;
            });
        };

        // project single sample => coordinates on components
        const project = (sample) => {
            const centered = sample.map((v, i) => v - mean[i]);
            return components.map(c => dot(centered, c));
        };

        return {
            components, // principal axes (k x nFeatures)
            explainedVariance,
            explainedVarianceRatio,
            mean,
            transform,
            inverseTransform,
            project
        };
    }
    /**
     * Independent Component Analysis (ICA)
     *
     * @param {Array.<Array.<number>>} data - 2D array with shape [nSamples][nFeatures]
     * @param {Object} [options] - Optional settings
     * @param {number} [options.nComponents] - Number of independent components to extract
     * @param {number} [options.maxIter] - Maximum number of iterations
     * @param {number} [options.tol] - Tolerance for convergence
     * @returns {{
     *   components: Array.<Array.<number>>,      // independent components (nComponents x nFeatures)
     *   transform: function(Array.<Array.<number>>): Array.<Array.<number>>,
     *   inverseTransform: function(Array.<Array.<number>>): Array.<Array.<number>>,
     *   project: function(Array.<number>): Array.<number>
     * }}
     */
    ICA(data, options = {}) {
        const nSamples = data.length;
        const nFeatures = data[0].length;
        const nComponents = options.nComponents || nFeatures;
        const maxIter = options.maxIter || 200;
        const tol = options.tol || 1e-4;

        function mean(matrix) {
            const meanArr = new Array(matrix[0].length).fill(0);
            for (let i = 0; i < matrix.length; i++) {
                for (let j = 0; j < matrix[0].length; j++) {
                    meanArr[j] += matrix[i][j];
                }
            }
            for (let j = 0; j < meanArr.length; j++) {
                meanArr[j] /= matrix.length;
            }
            return meanArr;
        }

        function center(data) {
            const m = mean(data);
            return data.map(row => row.map((val, idx) => val - m[idx]));
        }

        function dot(a, b) {
            return a.reduce((sum, val, i) => sum + val * b[i], 0);
        }

        function transpose(matrix) {
            const rows = matrix.length;
            const cols = matrix[0].length;
            const transposed = Array.from({ length: cols }, () => new Array(rows));
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    transposed[j][i] = matrix[i][j];
                }
            }
            return transposed;
        }

        function multiplyMatrices(A, B) {
            const aRows = A.length;
            const aCols = A[0].length;
            const bRows = B.length;
            const bCols = B[0].length;
            if (aCols !== bRows) throw new Error("Matrix dimensions do not match");
            const result = Array.from({ length: aRows }, () => new Array(bCols).fill(0));
            for (let i = 0; i < aRows; i++) {
                for (let j = 0; j < bCols; j++) {
                    for (let k = 0; k < aCols; k++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return result;
        }

        // Center the data
        const dataMean = mean(data);
        const X = center(data);

        // Initialize separation matrix
        let W = Array.from({ length: nComponents }, () => 
            Array.from({ length: nFeatures }, () => Math.random() - 0.5)
        );

        // FastICA iterations
        for (let iter = 0; iter < maxIter; iter++) {
            const W_old = JSON.parse(JSON.stringify(W));

            for (let i = 0; i < nComponents; i++) {
                const w = W[i];

                // Project data
                const wx = X.map(row => dot(row, w));

                // Apply contrast function (tanh)
                const gw = wx.map(val => Math.tanh(val));
                
                // Update W
                let wNew = Array.from({ length: nFeatures }, (_, j) =>
                    X.reduce((sum, row, n) => sum + row[j] * gw[n], 0) / nSamples
                );

                // Decorrelate
                for (let j = 0; j < i; j++) {
                    const dotProduct = dot(wNew, W[j]);
                    wNew = wNew.map((val, k) => val - dotProduct * W[j][k]);
                }

                // Normalize
                const normW = Math.sqrt(dot(wNew, wNew));
                W[i] = wNew.map(val => val / normW);
            }

            // Check convergence
            let maxDiff = 0;
            for (let i = 0; i < nComponents; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(dot(W[i], W_old[i]) - 1));
            }
            if (maxDiff < tol) break;
        }

        function transform(newData) {
            const centered = newData.map(row => row.map((val, idx) => val - dataMean[idx]));
            return multiplyMatrices(centered, transpose(W));
        }

        function project(sample) {
            const centeredSample = sample.map((val, idx) => val - dataMean[idx]);
            return multiplyMatrices([centeredSample], transpose(W))[0];
        }

        return {
            components: W,
            transform,
            inverseTransform: (components) => multiplyMatrices(components, W),
            project
        };
    }
    /**
     * t-Distributed Stochastic Neighbor Embedding (t-SNE)
     * @param {Array.<Array.<number>>} data - High-dimensional data points (array of points)
     * @param {Object} options - Configuration options
     * @returns {Array.<Array.<number>>} Embedded low-dimensional points
     */
    tSNE(data, options = {}) {
        const perplexity = options.perplexity || 30;
        const dim = options.dim || 2;
        const maxIter = options.maxIter || 1000;
        const learningRate = options.learningRate || 200;

        const n = data.length;

        // Step 1: Compute pairwise distances
        const distances = Array.from({ length: n }, () => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const dist = MLMath.distance(data[i], data[j], 'euclidean');
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        // Step 2: Compute sigmas via binary search for each point
        const sigmas = MLMath.binarySearchSigma(distances, perplexity);

        // Compute P affinities
        const P = Array.from({ length: n }, () => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    // Exponentiate with sigma
                    P[i][j] = Math.exp((-distances[i][j]) ** 2 / (2 * sigmas[i] ** 2));
                    sum += P[i][j];
                }
            }
            // Normalize
            for (let j = 0; j < n; j++) {
                P[i][j] /= sum;
            }
        }

        // Symmetrize P
        const P_sym = Array.from({ length: n }, () => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                P_sym[i][j] = (P[i][j] + P[j][i]) / (2 * n);
            }
        }

        // Initialize low-dimensional points randomly
        const Y = Array.from({ length: n }, () =>
            Array.from({ length: dim }, () => (Math.random() - 0.5) * 1e-4)
        );

        // Initialize velocities (dY) for gradient
        const dY = Array.from({ length: n }, () => Array(dim).fill(0));
        // Initialize gains
        const gains = Array.from({ length: n }, () => Array(dim).fill(1));
        let momentum = 0.5;

        // Gradient descent iterations
        for (let iter = 0; iter < maxIter; iter++) {
            // Compute pairwise affinities q_ij
            const Q = Array.from({ length: n }, () => Array(n).fill(0));
            let sumQ = 0;
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    const diff = MLMath.distance(Y[i], Y[j], 'euclidean');
                    const q = 1 / (1 + diff ** 2);
                    Q[i][j] = q;
                    Q[j][i] = q;
                    sumQ += 2 * q;
                }
            }
            // Normalize Q
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Q[i][j] /= sumQ;
                }
            }

            // Compute gradient
            for (let i = 0; i < n; i++) {
                for (let d = 0; d < dim; d++) {
                    let grad = 0;
                    for (let j = 0; j < n; j++) {
                        if (i !== j) {
                            const diff = Y[i][d] - Y[j][d];
                            const p_ij = P_sym[i][j];
                            const q_ij = Q[i][j];
                            // Gradient component
                            grad += (p_ij - q_ij) * diff * 4;
                        }
                    }
                    dY[i][d] = grad;
                }
            }

            // Update Y with momentum, gains, and gradient
            for (let i = 0; i < n; i++) {
                for (let d = 0; d < dim; d++) {
                    // Adjust gains
                    if (Math.sign(dY[i][d]) !== Math.sign(gains[i][d])) {
                        gains[i][d] += 0.2;
                    } else {
                        gains[i][d] *= 0.8;
                    }
                    if (gains[i][d] < 0.01) gains[i][d] = 0.01;

                    // Update velocity with momentum
                    dY[i][d] = momentum * dY[i][d] + (1 - momentum) * gains[i][d] * dY[i][d];

                    // Update positions
                    Y[i][d] += learningRate * dY[i][d];
                }
            }

            // Anneal momentum
            if (iter === 250) momentum = 0.8;
            if (iter === 500) momentum = 0.9;
        }

        return Y;
    }
    /**
     * Density-Based Spatial Clustering of Applications with Noise
     * @param {number} eps - The maximum distance for two points to be considered neighbors.
     * @param {number} minPts - Minimum number of points required to form a dense region (cluster).
     * @param {number[][]} points Array of data points to cluster.
     */
    DBScan(eps, minPts, points){
        const dbs = new DBScan(eps, minPts)
        return dbs.fit(points);
    }

    /**
     * Decomposes a time series into its trend, seasonal, and residual components.
     * Uses moving average for trend estimation and seasonal averaging for seasonal component.
     * 
     * @param {number[]} data - The input time series data array.
     * @param {number} [seasonalPeriod=12] - The seasonal period (e.g., 12 for monthly data).
     * @returns {Object} An object containing the trend, seasonal, and residual components:
     *  - trend: Number[] - Estimated trend component.
     *  - seasonal: Number[] - Seasonal component aligned with data.
     *  - residual: Number[] - Residual component (data - trend - seasonal).
     * 
     * @throws Will throw an error if data is not an array or its length is less than seasonalPeriod.
     */
    DecomposeTimeSeries(data, seasonalPeriod = 12){
        if (!Array.isArray(data) || data.length < seasonalPeriod) 
            throw new Error("Data must be an array with length at least equal to the seasonal period");
        

        const n = data.length;

        // Step 1: Estimate the trend component using a moving average
        const trend = [];
        const windowSize = seasonalPeriod;

        for (let i = 0; i < n; i++) {
            let start = Math.max(0, i - Math.floor(windowSize / 2));
            let end = Math.min(n - 1, i + Math.floor(windowSize / 2));
            let sum = 0;
            let count = 0;
            for (let j = start; j <= end; j++) {
            sum += data[j];
            count++;
            }
            trend.push(sum / count);
        }

        // Step 2: Detrend the data
        const detrended = data.map((val, i) => val - trend[i]);

        // Step 3: Estimate seasonal component
        const seasonalSums = new Array(seasonalPeriod).fill(0);
        const seasonalCounts = new Array(seasonalPeriod).fill(0);

        for (let i = 0; i < n; i++) {
            const seasonIndex = i % seasonalPeriod;
            seasonalSums[seasonIndex] += detrended[i];
            seasonalCounts[seasonIndex] += 1;
        }

        const seasonalIndices = seasonalSums.map((sum, i) => sum / seasonalCounts[i]);

        // Step 4: Construct seasonal component aligned with data
        const seasonalComponent = [];
        for (let i = 0; i < n; i++) {
            const seasonIndex = i % seasonalPeriod;
            seasonalComponent.push(seasonalIndices[seasonIndex]);
        }

        // Step 5: Calculate residuals
        const residual = data.map((val, i) => val - trend[i] - seasonalComponent[i]);

        return {
            trend: trend,
            seasonal: seasonalComponent,
            residual: residual,
        };
    }
}

