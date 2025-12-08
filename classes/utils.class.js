
/**
 * Datasets Utilities
 *
 * @class Dataset Util
 */
export const Dataset = class{
    construct(){

    }
    /**
     * Merges multiple datasets (arrays) into a single array of arrays,
     * where each inner array contains elements from each dataset at the same index.
     *
     * @param {...Array} datasets - The datasets to merge. All datasets must have the same length.
     * @returns {Array} An array of merged data points, each being an array of corresponding elements.
     *
     * @throws {Error} If the datasets have different lengths.
     *
     * @example
     * const x1 = [1, 2, 3];
     * const x2 = [4, 5, 6];
     * const x3 = [7, 8, 9];
     * const result = mergeDatasets(x1, x2, x3);
     * //result: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
     */
    merge(...datasets){
        if (datasets.length === 0) return [];
        const length = datasets[0].length;
        for (let i = 1; i < datasets.length; i++) {
            if (datasets[i].length !== length) {
            throw new Error('All datasets must have the same length');
            }
        }
        const merged = [];
        for (let i = 0; i < length; i++) {
            const combined = datasets.map(dataset => dataset[i]);
            merged.push(combined);
        }
        return merged;
    }
    /**
     * Loads a CSV file from the given file path and converts it into an array.
     * 
     * @param {string} file_path - The path to the CSV file.
     * @param {Object} options - Options for parsing the CSV.
     * @returns {string} Returns a promise
     * @example
     *        
     */
    
    /**
     * Loads a CSV file from the given file path and converts it into an array.
     *
     * @async
     * @param {string} file_path - File path to the CSV file
     * @param {{delimiter: string, header: boolean, skipEmptyLines: boolean}} [options={}] - Options for parsing the CSV.
     * @example
     * {
     *  delimiter: ',',          // delimiter used in CSV (default is comma)
     *  header: true,            // whether the first row contains headers
     *  skipEmptyLines: true,    // whether to skip empty lines
     *  newline: '\n'            // Split each row with a newline
     * }
     * @returns {Promise<Array>} Returns an array from the CSV
     */
    async load_file(file_path, options = {}) {
        // Set default options
        const {
            delimiter = ',',
            header = false,
            skipEmptyLines = false,
            newline = '\n'
        } = options;

        // Fetch the CSV file
        const response = await fetch(file_path);
        const csvText = await response.text();

        // Split into lines based on newline option
        const lines = csvText.split(newline);

        // Process lines based on options
        const data = [];

        // If header is true, extract headers
        let headers = [];
        let startIndex = 0;
        if (header && lines.length > 0) {
            headers = lines[0].split(delimiter);
            startIndex = 1;
        }

        for (let i = startIndex; i < lines.length; i++) {
            const line = lines[i].trim();
            if (skipEmptyLines && line === '') continue;

            const values = line.split(delimiter);

            if (header) {
                const obj = {};
                headers.forEach((h, index) => {
                    obj[h] = values[index];
                });
                data.push(obj);
            } else {
                data.push(values);
            }
        }

        return data;
    }
    /**
     * Retrieves the label from the dataset corresponding to the predicted index.
     * @param {Array<Object>} data - The dataset array of objects, each with features and a label.
     * @param {number} predictedInd - The index of the predicted class or label.
     * @returns {string|number} - The label (string or number) at the specified index in the dataset.
     */
    getLabel(data, predictedInd){
        if (!Array.isArray(data) || predictedInd < 0 || predictedInd >= data.length)
            throw new Error("Data isn't valid");
        return data[predictedInd].label
    }
};


/**
 * Generate random data suitable for binary SVM classification (2 classes).
 *
 * @param {number} sampleSize - Number of samples
 * @param {number} featureSize - Number of features per sample
 * @param {number} [min=0] - Minimum feature value
 * @param {number} [max=1] - Maximum feature value
 * @returns {{features: Array<Array<number>>, labels: number[]}}
 */
export const randomData = (sampleSize = 2, featureSize = 2, min = 0, max = 1) => {
    const generated = {
        features: [],
        labels: []
    };
    for (let i = 0; i < sampleSize; i++) {
        // Generate an array of random floats for features
        const featureSample = [];
        for (let j = 0; j < featureSize; j++) {
            featureSample.push(Math.random() * (max - min) + min); // random float between min and max
        }
        generated['features'].push(featureSample);
        
        // Assign label 0 or 1 randomly to ensure exactly 2 classes
        if(sampleSize==2){
            let random = Math.round(Math.random());
            while(generated.labels.indexOf(random)!=-1){
                random = Math.round(Math.random());
            }
            generated['labels'].push(random);
        }else
            generated['labels'].push(Math.round(Math.random())); // 0 or 1
    }
    return generated;
};

/**
 * Scales features and label values of a dataset to the 0-1 range.
 *
 * @param {Object[]} dataset - Array of data objects to be scaled.
 * @param {string} label - The key of the label property in each data object.
 * @returns {Object} An object containing:
 *   - {Object[]} scaledData: The dataset with scaled feature and label values.
 *   - {Object} featureStats: Min and max values for each feature.
 *   - {Object} labelStats: Min and max values for the label.
 */
export const ScaleData = (dataset, label) => {
    if (!Array.isArray(dataset) || dataset.length === 0) return [];
    

    // Gather all feature keys (excluding label)
    const featureKeys = Object.keys(dataset[0]).filter(k => k !== label);

    // Initialize min/max for each feature
    const featureStats = {};
    featureKeys.forEach(key => {
        featureStats[key] = { min: Infinity, max: -Infinity };
    });

    // Find min and max for features and label
    let labelMin = Infinity;
    let labelMax = -Infinity;

    dataset.forEach(item => {
        featureKeys.forEach(key => {
        const val = Number(item[key]);
        if (val < featureStats[key].min) featureStats[key].min = val;
        if (val > featureStats[key].max) featureStats[key].max = val;
        });
        const labelVal = Number(item[label]);
        if (labelVal < labelMin) labelMin = labelVal;
        if (labelVal > labelMax) labelMax = labelVal;
    });

    // Function to scale a value
    const scaleValue = (val, min, max) => {
        if (max === min) return 0; // avoid division by zero
        return (val - min) / (max - min);
    };

    // Create scaled dataset
    const scaledData = dataset.map(item => {
        const scaledItem = {};

        featureKeys.forEach(key => {
        scaledItem[key] = scaleValue(Number(item[key]), featureStats[key].min, featureStats[key].max);
        });

        // Scale label
        scaledItem[label] = scaleValue(Number(item[label]), labelMin, labelMax);

        return scaledItem;
    });

    return {
        scaledData,
        featureStats, // optional: store min/max info for inverse scaling if needed
        labelStats: { min: labelMin, max: labelMax },
    };
};