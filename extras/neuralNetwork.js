export const NeuralNetwork = class {
  constructor(data = [], features = [], options = {}) {
    this.data = data || [];
    this.features = features || [];

    const {
      hiddenLayers = 1,
      epoch = 200,
      learningRate = 0.05,
      activation = 'ReLU',
      regularization = 'none',
      noise = 0,
      batch = 15,
      ratioTrainTest = 0.8,
      type = 'classification',
      debug = false,
      labelKey = 'label',
    } = options;

    // configuration
    this.hiddenLayers = Array.isArray(hiddenLayers) ? hiddenLayers : new Array(Number(hiddenLayers)).fill(8);
    this.epoch = Number(epoch) || 200;
    this.learningRate = Number(learningRate) || 0.05;
    this.activation = String(activation || 'ReLU');
    this.regularization = String(regularization || 'none');
    this.noise = Number(noise) || 0;
    this.batch = Math.max(1, Number(batch) || 15);
    this.ratioTrainTest = Number(ratioTrainTest) || 0.8;
    this.type = type === 'regression' ? 'regression' : 'classification';
    this.debug = !!debug;
    this.labelKey = labelKey;

    // internal
    this.encoders = {}; // for categorical features -> one-hot sizes and maps
    this.classes = []; // label classes for classification
    this.inputSize = 0;
    this.outputSize = this.type === 'classification' ? 0 : 1;
    this.weights = []; // weight matrices
    this.biases = []; // bias vectors

    if (this.data && this.data.length && this.features && this.features.length) {
      this._prepare();
      this._initNetwork();
    }
  }

  // prepare encoders, determine input/output sizes and split data
  _prepare() {
    // build encoders for categorical features and determine input size
    this.encoders = {};
    this.inputSize = 0;

    // detect categorical feature values from data
    for (const feat of this.features) {
      let isCategorical = false;
      for (const row of this.data) {
        const v = row[feat];
        if (typeof v === 'string' || typeof v === 'boolean') {
          isCategorical = true;
          break;
        }
      }
      if (isCategorical) {
        // build value list
        const map = {};
        let idx = 0;
        for (const row of this.data) {
          const val = row[feat];
          if (!(val in map)) {
            map[val] = idx++;
          }
        }
        this.encoders[feat] = { type: 'onehot', map, size: Object.keys(map).length };
        this.inputSize += this.encoders[feat].size;
      } else {
        // treat as numeric
        this.encoders[feat] = { type: 'numeric', size: 1 };
        this.inputSize += 1;
      }
    }

    // handle labels
    if (this.type === 'classification') {
      const classSet = new Map();
      for (const row of this.data) {
        const label = row[this.labelKey];
        if (!classSet.has(label)) classSet.set(label, classSet.size);
      }
      this.classes = Array.from(classSet.keys());
      this.outputSize = this.classes.length || 1;
    } else {
      this.outputSize = 1;
    }

    // train/test split
    const shuffled = this.data.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const split = Math.floor(shuffled.length * this.ratioTrainTest);
    this.trainData = shuffled.slice(0, split);
    this.testData = shuffled.slice(split);
  }

  // initialize weight matrices and biases
  _initNetwork() {
    const layerSizes = [this.inputSize, ...this.hiddenLayers, this.outputSize];
    this.weights = [];
    this.biases = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const inSize = layerSizes[i];
      const outSize = layerSizes[i + 1];
      // Xavier/He init depending on activation; keep simple random
      const scale = Math.sqrt(2 / Math.max(1, inSize));
      const w = new Array(outSize).fill(0).map(() =>
        new Array(inSize).fill(0).map(() => (Math.random() * 2 - 1) * scale)
      );
      const b = new Array(outSize).fill(0);
      this.weights.push(w);
      this.biases.push(b);
    }
  }

  // encode a single sample into numeric input vector
  _encodeSample(sample) {
    const x = [];
    for (const feat of this.features) {
      const encoder = this.encoders[feat];
      const val = sample[feat];
      if (encoder.type === 'onehot') {
        const one = new Array(encoder.size).fill(0);
        const idx = encoder.map.hasOwnProperty(val) ? encoder.map[val] : -1;
        if (idx >= 0) one[idx] = 1;
        x.push(...one);
      } else {
        const num = Number(val) || 0;
        x.push(num);
      }
    }
    return x;
  }

  // activation functions and derivatives
  // activation functions
_activate(z) {
  if (this.activation === 'sigmoid') {
    return z.map(v => 1 / (1 + Math.exp(-v)));
  } else if (this.activation === 'tanh') {
    return z.map(v => Math.tanh(v));
  } else if (this.activation === 'linear') {
    return z.slice(); // linear just returns input
  }
  // default ReLU
  return z.map(v => Math.max(0, v));
}
_activatePrime(z) {
  if (this.activation === 'sigmoid') {
    const s = z.map(v => 1 / (1 + Math.exp(-v)));
    return s.map(v => v * (1 - v));
  } else if (this.activation === 'tanh') {
    return z.map(v => 1 - Math.tanh(v) * Math.tanh(v));
  } else if (this.activation === 'linear') {
    return new Array(z.length).fill(1); // derivative of linear is 1
  }
  // default ReLU prime
  return z.map(v => (v > 0 ? 1 : 0));
}
  _softmax(z) {
    const max = Math.max(...z);
    const exps = z.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map(e => e / sum);
  }

  // forward pass for one sample, returns { activations, zs }
  _forward(xRaw) {
    let a = xRaw.slice();
    const activations = [a];
    const zs = [];
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l];
      const b = this.biases[l];
      const z = new Array(w.length).fill(0).map((_, i) => {
        let sum = b[i];
        for (let j = 0; j < w[i].length; j++) {
          sum += w[i][j] * a[j];
        }
        return sum;
      });
      zs.push(z);
      if (l === this.weights.length - 1 && this.type === 'classification') {
        a = this._softmax(z);
      } else {
        a = this._activate(z);
      }
      activations.push(a);
    }
    return { activations, zs };
  }

  // compute loss and gradients for one batch, update weights
  _updateBatch(batch) {
    // accumulators
    const nablaW = this.weights.map(w => w.map(row => new Array(row.length).fill(0)));
    const nablaB = this.biases.map(b => new Array(b.length).fill(0));

    for (const sample of batch) {
      const x = this._encodeSample(sample);
      const { activations, zs } = this._forward(x);

      // expected output
      let y;
      if (this.type === 'classification') {
        y = new Array(this.outputSize).fill(0);
        const idx = this.classes.indexOf(sample[this.labelKey]);
        if (idx >= 0) y[idx] = 1;
      } else {
        y = [Number(sample[this.labelKey]) || 0];
      }

      // backpropagate
      // delta for last layer
      const L = this.weights.length - 1;
      let delta = new Array(this.outputSize).fill(0);
      if (this.type === 'classification') {
        // cross-entropy with softmax simplifies to (a - y)
        const aL = activations[activations.length - 1];
        for (let i = 0; i < delta.length; i++) delta[i] = aL[i] - y[i];
      } else {
        // regression MSE: delta = (a - y) * activation'
        const aL = activations[activations.length - 1];
        const zL = zs[zs.length - 1];
        const aprime = this._activatePrime(zL);
        for (let i = 0; i < delta.length; i++) delta[i] = (aL[i] - y[i]) * aprime[i];
      }

      // accumulate gradients for last layer
      const aPrev = activations[activations.length - 2];
      for (let i = 0; i < delta.length; i++) {
        nablaB[L][i] += delta[i];
        for (let j = 0; j < aPrev.length; j++) {
          nablaW[L][i][j] += delta[i] * aPrev[j];
        }
      }

      // propagate through previous layers
      for (let l = L - 1; l >= 0; l--) {
        const z = zs[l];
        const sp = this._activatePrime(z);
        const wNext = this.weights[l + 1];
        const deltaNext = delta;
        const deltaCurr = new Array(this.weights[l].length).fill(0);
        for (let i = 0; i < this.weights[l].length; i++) {
          let sum = 0;
          for (let k = 0; k < deltaNext.length; k++) {
            sum += wNext[k][i] * deltaNext[k];
          }
          deltaCurr[i] = sum * sp[i];
        }
        delta = deltaCurr;
        const aPrevL = activations[l];
        for (let i = 0; i < delta.length; i++) {
          nablaB[l][i] += delta[i];
          for (let j = 0; j < aPrevL.length; j++) {
            nablaW[l][i][j] += delta[i] * aPrevL[j];
          }
        }
      }
    }

    // apply gradients
    const eta = this.learningRate / batch.length;
    for (let l = 0; l < this.weights.length; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          // L2 regularization
          let regTerm = 0;
          if (this.regularization === 'l2') regTerm = (this.weights[l][i][j] * 0.0001);
          // L1 regularization
          if (this.regularization === 'l1') regTerm = Math.sign(this.weights[l][i][j]) * 0.0001;
          this.weights[l][i][j] -= eta * (nablaW[l][i][j] + regTerm);
        }
        this.biases[l][i] -= eta * nablaB[l][i];
      }
    }
  }

  // fit/train the network
  fit(options = {}) {
    const epochs = options.epoch || this.epoch;
    const batchSize = options.batch || this.batch;

    if (!this.trainData || !this.trainData.length) return;

    for (let e = 0; e < epochs; e++) {
      // shuffle training data each epoch
      for (let i = this.trainData.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [this.trainData[i], this.trainData[j]] = [this.trainData[j], this.trainData[i]];
      }

      // create batches
      for (let i = 0; i < this.trainData.length; i += batchSize) {
        const batch = this.trainData.slice(i, i + batchSize);
        // optional noise augmentation for inputs
        if (this.noise > 0) {
          for (const s of batch) {
            for (const feat of this.features) {
              if (this.encoders[feat].type === 'numeric') {
                s[feat] = Number(s[feat]) + (Math.random() * 2 - 1) * this.noise;
              }
            }
          }
        }
        this._updateBatch(batch);
      }

      if (this.debug && (e % Math.max(1, Math.floor(epochs / 5)) === 0)) {
        const metric = this.evaluate();
        console.log(`Epoch ${e + 1}/${epochs} - ${JSON.stringify(metric)}`);
      }
    }
  }

  // predict for a single sample, returns label or number
  predict(sample) {
    const x = this._encodeSample(sample);
    const { activations } = this._forward(x);
    const out = activations[activations.length - 1];

    if (this.type === 'classification') {
      const idx = out.indexOf(Math.max(...out));
      return this.classes[idx];
    }
    return out[0];
  }

  // evaluate on test set: returns accuracy for classification or mse for regression
  evaluate() {
    if (!this.testData || !this.testData.length) 
      return null; // No test data to evaluate
    if (this.type === 'classification') {
      let correct = 0;
      for (const sample of this.testData) {
        const predictedLabel = this.predict(sample);
        const trueLabel = sample[this.labelKey];
        if (predictedLabel === trueLabel) correct++;
      }
      const accuracy = correct / this.testData.length;
      return { accuracy };
    } else if (this.type === 'regression') {
      let sumSquaredError = 0;
      for (const sample of this.testData) {
        const predicted = this.predict(sample);
        const trueVal = Number(sample[this.labelKey]);
        const error = predicted - trueVal;
        sumSquaredError += error * error;
      }
      const mse = sumSquaredError / this.testData.length;
      return { mse };
    } else {
      return null; // Unknown type
    }
  }

  // convenience: train on provided data (re-prepare and re-init)
  trainOn(data, features, options = {}) {
    this.data = data.slice();
    this.features = features.slice();
    Object.assign(this, options); // allow overriding simple top-level opts if provided
    this._prepare();
    this._initNetwork();
    this.fit(options);
  }
  // for regression: get actual data point closest to predicted value
  getActual(scaled, key='label') {
    const min = Math.min(...this.data.map(d=>d[key])),
          max = Math.max(...this.data.map(d=>d[key]));
    if(this.type==='regression'){
      const predicted = scaled*(max - min) + min;
      // Find closest index
      const closestIndex = this.data.reduce((prevIdx, current, idx) => {
        const prevDiff = Math.abs(this.data[prevIdx][key] - predicted);
        const currentDiff = Math.abs(current[key] - predicted);
        return currentDiff < prevDiff ? idx : prevIdx;
      }, 0);
      return this.data[closestIndex][key]??0;
    }
  }

};
