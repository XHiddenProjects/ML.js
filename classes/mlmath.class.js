export const MLMath = {
    ...Math,
    /**
     * Get the accuracy of the true label and predicted labels
     * @param {Number[]} trueLabel True labels
     * @param {Number[]} predictedLabels Predicted labels
     * @returns {Number} returns the percentage
     */
    accuracy: (trueLabel, predictedLabels)=>{
        if (trueLabel.length !== predictedLabels.length) 
            throw new Error("Input arrays must be of the same length");
        // Calculate total squared error
        let totalError = 0;
        for (let i = 0; i < trueLabel.length; i++) {
            let diff = trueLabel[i] - predictedLabels[i];
            totalError += diff * diff;
        }
        // Define maximum possible total error for normalization
        const maxErrorPerComponent = 2; // max difference between -1 and 1
        const maxTotalError = maxErrorPerComponent * trueLabel.length;

        // Calculate accuracy as a percentage
        let accuracy = Math.max(0, 1 - (totalError / maxTotalError)); 
        return (accuracy * 100)
    },
    /**
     * Calculate distance between two points
     * @param {Number|Number[]} pointA Point A
     * @param {Number|Number[]} pointB Point B
     * @param {'euclidean'|'manhattan'|'minkowski'|'chebyshev'|'cosine'|'hamming'|'jaccard'|'bray-curtis'|'mahalanobis'|'canberra'|'correlation'|'dice'|'dice-sorensen'|'bhattacharyya'|'wasserstein'|'levenshtein'|'haversine'|'jensen-shannon'} metric Distance metric
     * @param {{p:number, invCovMatrix: number[]}} options Additional options
     * @returns {Number} Distance value
     */
    distance: (pointA, pointB, metric = 'euclidean',options={})=>{
        metric = metric.toLowerCase();
        let sum = 0;
        if (metric === 'euclidean') {
            for (let i = 0; i < pointA.length; i++) sum += Math.pow(pointA[i] - pointB[i], 2);
            return Math.sqrt(sum);
        } else if (metric === 'manhattan') {
            for (let i = 0; i < pointA.length; i++) sum += Math.abs(pointA[i] - pointB[i]);
            return sum;
        } else if(metric==='minkowski'){
            const p = options.p||3; // You can parameterize this value as needed
            for (let i = 0; i < pointA.length; i++) sum += Math.pow(Math.abs(pointA[i] - pointB[i]), p);
            return Math.pow(sum, 1 / p);
        }else if(metric==='chebyshev'){
            let maxDiff = 0;
            for (let i = 0; i < pointA.length; i++) {
                const diff = Math.abs(pointA[i] - pointB[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            return maxDiff;
        }else if(metric==='cosine'){
            let dotProduct = 0,
            normA = 0,
            normB = 0;
            for (let i = 0; i < pointA.length; i++) {
                dotProduct += pointA[i] * pointB[i];
                normA += pointA[i] * pointA[i];
                normB += pointB[i] * pointB[i];
            }
            if (normA === 0 || normB === 0) {
                throw new Error("Cannot compute cosine distance for zero-length vectors");
            }
            return 1 - (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
        }else if(metric==='hamming'){
            let differingComponents = 0;
            for (let i = 0; i < pointA.length; i++) {
                if (pointA[i] !== pointB[i]) {
                    differingComponents++;
                }
            }
            return differingComponents / pointA.length;
        }else if(metric==='jaccard'){
            let intersection = 0;
            let union = 0;
            for (let i = 0; i < pointA.length; i++) {
                if (pointA[i] === 1 || pointB[i] === 1) {
                    union++;
                    if (pointA[i] === 1 && pointB[i] === 1) intersection++;
                }
            }
            if (union === 0) return 0; // Both sets are empty
            return 1 - (intersection / union);
        }else if(metric==='bray-curtis'){
            let numerator = 0;
            let denominator = 0;
            for (let i = 0; i < pointA.length; i++) {
                numerator += Math.abs(pointA[i] - pointB[i]);
                denominator += Math.abs(pointA[i] + pointB[i]);
            }
            if (denominator === 0) return 0; // Both points are zero vectors
            
            return numerator / denominator;
        }else if(metric==='mahalanobis'){
            const invCovMatrix = options.invCovMatrix||null;
            if (!invCovMatrix) throw new Error("Inverse covariance matrix is required for Mahalanobis distance");
            const diff = pointA.map((val, idx) => val - pointB[idx]);
            let leftProduct = new Array(diff.length).fill(0);
            for (let i = 0; i < invCovMatrix.length; i++) {
                for (let j = 0; j < invCovMatrix[i].length; j++) leftProduct[i] += diff[j] * invCovMatrix[j][i];
            }
            let mahalanobisDistance = 0;
            for (let i = 0; i < diff.length; i++) mahalanobisDistance += leftProduct[i] * diff[i];
            
            return Math.sqrt(mahalanobisDistance);
        }else if(metric==='canberra'){
            let sum = 0;
            for (let i = 0; i < pointA.length; i++) {
                const numerator = Math.abs(pointA[i] - pointB[i]),
                denominator = Math.abs(pointA[i]) + Math.abs(pointB[i]);
                if (denominator !== 0) sum += numerator / denominator;
            }
            return sum;
        }else if(metric==='correlation'){
            const meanA = pointA.reduce((acc, val) => acc + val, 0) / pointA.length,
            meanB = pointB.reduce((acc, val) => acc + val, 0) / pointB.length;
            let numerator = 0,
            denomA = 0,
            denomB = 0;
            for (let i = 0; i < pointA.length; i++) {
                const diffA = pointA[i] - meanA;
                const diffB = pointB[i] - meanB;
                numerator += diffA * diffB;
                denomA += diffA * diffA;
                denomB += diffB * diffB;
            }
            const denominator = Math.sqrt(denomA) * Math.sqrt(denomB);
            if (denominator === 0) throw new Error("Cannot compute correlation distance for zero-variance vectors");
            const correlation = numerator / denominator;
            return 1 - correlation;
        }else if(metric==='dice'||metric==='dice-sorensen'){
            let intersection = 0,
            sizeA = 0,
            sizeB = 0;
            for (let i = 0; i < pointA.length; i++) {
                if (pointA[i] === 1) sizeA++;
                if (pointB[i] === 1) sizeB++;
                if (pointA[i] === 1 && pointB[i] === 1) intersection++;
            }
            const diceCoefficient = (2 * intersection) / (sizeA + sizeB);
            return 1 - diceCoefficient;
        }else if(metric==='bhattacharyya'){
            let coeff = 0;
            for (let i = 0; i < pointA.length; i++) coeff += Math.sqrt(pointA[i] * pointB[i]);
            return -Math.log(coeff);
        }else if(metric==='wasserstein'){
            const sortedA = [...pointA].sort((a, b) => a - b),
            sortedB = [...pointB].sort((a, b) => a - b);
            let sum = 0;
            for (let i = 0; i < sortedA.length; i++) sum += Math.abs(sortedA[i] - sortedB[i]);
            return sum / sortedA.length;
        }else if(metric==='levenshtein'){
            const aLen = pointA.length,
            bLen = pointB.length,
            dp = Array.from({ length: aLen + 1 }, () => new Array(bLen + 1).fill(0));
            for (let i = 0; i <= aLen; i++) dp[i][0] = i;
            for (let j = 0; j <= bLen; j++) dp[0][j] = j;
            for (let i = 1; i <= aLen; i++) {
                for (let j = 1; j <= bLen; j++) {
                    const cost = pointA[i - 1] === pointB[j - 1] ? 0 : 1;
                    dp[i][j] = Math.min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost
                    );
                }
            }
            return dp[aLen][bLen];
        }else if(metric==='haversine'){
            const toRadians = (degree) => degree * (Math.PI / 180),
            lat1 = toRadians(pointA[0]),
            lon1 = toRadians(pointA[1]),
            lat2 = toRadians(pointB[0]),
            lon2 = toRadians(pointB[1]),
            dLat = lat2 - lat1,
            dLon = lon2 - lon1,
            a = Math.sin(dLat / 2) ** 2 +
                      Math.cos(lat1) * Math.cos(lat2) *
                      Math.sin(dLon / 2) ** 2,
            c = 2 * Math.asin(Math.sqrt(a)),
            R = 6371; // Radius of Earth in kilometers
            return R * c;
        }else if(metric==='jensen-shannon'){
            const klDivergence = (P, Q) => {
                let sum = 0;
                for (let i = 0; i < P.length; i++) {
                    if (P[i] === 0) continue;
                    sum += P[i] * Math.log(P[i] / Q[i]);
                }
                return sum;
            };
            const M = pointA.map((val, idx) => (val + pointB[idx]) / 2);
            const jsDiv = (klDivergence(pointA, M) + klDivergence(pointB, M)) / 2;
            return Math.sqrt(jsDiv);
        }else throw new Error("Unsupported metric: " + metric);
    },
    /**
     * Performs binary search to find the optimal sigma values for each data point to match a target perplexity.
     * @param {Number[][]} distances - A 2D array where each element distances[i] is an array of distances from point i to all other points.
     * @param {Number} targetPerplexity - The desired perplexity value (roughly the effective number of neighbors).
     * @param {Number} [tol=1e-5] - Tolerance for the difference between the computed entropy and log(perplexity). Default is 1e-5.
     * @param {Number} [maxIter=50] - Maximum number of iterations for the binary search per point. Default is 50.
     * @returns {Number[]} - An array of sigma values corresponding to each data point.
     */
    binarySearchSigma: (distances, targetPerplexity, tol = 1e-5, maxIter = 50)=>{
        const n = distances.length;
        const logU = Math.log(targetPerplexity);
        const sigmas = new Array(n).fill(1);
        for (let i = 0; i < n; i++) {
            let sigmaMin = 1e-20;
            let sigmaMax = 1e20;
            let sigma = sigmas[i];
            for (let iter = 0; iter < maxIter; iter++) {
                let sumP = 0;
                for (let j = 0; j < n; j++) {
                    if (i !== j) {
                        const val = Math.exp(-distances[i][j] / (2 * sigma * sigma));
                        sumP += val;
                    }
                }
                const H = Math.log(sumP) + (distances[i].reduce((acc, d, j) => j !== i ? acc + (d * Math.exp(-d / (2 * sigma * sigma))) : acc, 0) / (2 * sigma * sigma * sumP));
                const Hdiff = H - logU;
                if (Math.abs(Hdiff) < tol) break;
                if (Hdiff > 0) {
                    sigmaMin = sigma;
                    sigma = (sigmaMax === 1e20) ? sigma * 2 : (sigma + sigmaMax) / 2;
                } else {
                    sigmaMax = sigma;
                    sigma = (sigmaMin === 1e-20) ? sigma / 2 : (sigma + sigmaMin) / 2;
                }
            }
            sigmas[i] = sigma;
        }
        return sigmas;
    },
    /**
     * Shuffle an array in place
     * @param {*[]} array Array to shuffle
     * @returns {*[]} Shuffled array
     */
    shuffleArray: (array)=>{
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    },
    /**
     * Generate a random integer between min and max
     * @param {Number} min Minimum
     * @param {Number} max Maximum
     * @param {Boolean} include Include max value
     * @returns {Number} Random integer
     */
    randomNumber: (min, max, include=false)=>{
        if(include) return Math.floor(Math.random() * (max - min + 1)) + min;
        else return Math.floor(Math.random() * (max - min)) + min;
    },
    /**
     * Generate a random number following normal distribution
     * @param {Number} mean Mean
     * @param {Number} stdDev Standard Deviation
     * @returns {Number} Random number
     */
    randomNormal: (mean=0, stdDev=1)=>{
        let u1 = 0, u2 = 0;
        while (u1 === 0) u1 = Math.random();
        while (u2 === 0) u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return z0 * stdDev + mean;
    },
    /**
     * Perform a permutation test to compare two groups
     * @param {Number[]} groupA First group
     * @param {Number[]} groupB Second group
     * @param {Number} numPermutations Number of permutations
     * @returns {Number} p-value
     */
    permutationTest(groupA, groupB, numPermutations=1000){
        const observedDiff = Math.abs(mean(groupA) - mean(groupB));
        const combined = groupA.concat(groupB);
        let count = 0;
        for (let i = 0; i < numPermutations; i++) {
            const shuffled = this.shuffleArray(combined.slice());
            const newGroupA = shuffled.slice(0, groupA.length);
            const newGroupB = shuffled.slice(groupA.length);
            const newDiff = Math.abs(mean(newGroupA) - mean(newGroupB));
            if (newDiff >= observedDiff) count++;
        }
        return (count + 1) / (numPermutations + 1); // p-value
    },
    /**
     * Bootstrap confidence interval for the mean
     * @param {*[]} data Data array
     * @param {Number} [numSamples=1000] Number of bootstrap samples
     * @param {Number} [confidenceLevel=0.95] Confidence level
     * @returns 
     */
    bootstrapConfidenceInterval(data, numSamples=1000, confidenceLevel=0.95){
        const means = [];
        const n = data.length;
        for (let i = 0; i < numSamples; i++) {
            const sample = [];
            for (let j = 0; j < n; j++) {
                const randIndex = Math.floor(Math.random() * n);
                sample.push(data[randIndex]);
            }
            const sampleMean = sample.reduce((acc, val) => acc + val, 0) / n;
            means.push(sampleMean);
        }
        means.sort((a, b) => a - b);
        const lowerIndex = Math.floor((1 - confidenceLevel) / 2 * numSamples);
        const upperIndex = Math.floor((1 + confidenceLevel) / 2 * numSamples);
        return {
            lower: means[lowerIndex],
            upper: means[upperIndex]
        };
    },
    /**
     * Calculates the scores in a softmax function
     * @param {Number[]} scores Scores to create a function 
     * @returns {Number}
     */
    Softmax: (scores)=>{
        const maxScore = Math.max(...scores);
        const expScores = scores.map((s) => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        return expScores.map((e) => e / sumExp);
    }
}