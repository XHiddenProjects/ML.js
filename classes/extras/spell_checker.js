import { MLMath } from "../classes/mlmath.class.js";
/**
 * 
 * @param {String} language Language
 * @param {Object[]} dictionary 
 */
export const SpellChecker = class {
    
    /**
     * @constructor Creates an instance of SpellChecker.
     * @param {string} [language='en'] Language code
     * @param {string} [country='us'] Country code
     */
    constructor(language='en',country='us') {
        this.language = `${language.toLowerCase().substring(0,2)}-${country.toLowerCase().substring(0,2)}` || 'en-us';
        /**
         * word: string{
         *  definition: string
         *  synonyms: [string]
         *  antonyms: [string]
         *  examples: string|[string]
         *  frequency: number
         *  partOfSpeech: string
         *  pronunciation: string
         *  etymology: string
         * }
         */
        this.dictionary = new Set();
    }
    
    /**
     * Add a word to the dictionary
     *
     * @param {...{word: string,definition: string, synonyms: string[], antonyms: string[], examples: string|string[], partOfSpeech: string, pronunciation: string, etymology: string}} wordObj 
     */
    addWord(wordObj) {
        if (wordObj && wordObj.word) {
            wordObj.frequency = 0;
            this.dictionary.add(wordObj);
        } else throw new Error("Invalid word object");
    }
    
    /**
     * Remove a word from the dictionary
     *
     * @param {String} word Word to remove
     */
    removeWord(word) {
        for (let entry of this.dictionary) {
            if (entry.word === word) {
                this.dictionary.delete(entry);
                return;
            }
        }
        throw new Error("Word not found in dictionary");
    }
    
    /**
     * Spell check a word
     *
     * @param {string} misspelledWord Misspelled word
     * @param {number} [maxSuggestions=5] Number of misspell suggestions to return
     * @returns {string[]} Array of suggested corrections
     */
    suggestCorrections(misspelledWord, maxSuggestions = 5) {
        const suggestions = [];
        for (let entry of this.dictionary) {
            const distance = MLMath.distance(misspelledWord, entry.word, 'levenshtein');
            // limit suggestions to those within a reasonable distance
            if (distance <= Math.max(1, Math.floor(misspelledWord.length / 3)))
                suggestions.push({ word: entry.word, distance });
        }
        suggestions.sort((a, b) => a.distance - b.distance);
        return suggestions.slice(0, maxSuggestions).map(s => s.word);
    }
    
    /**
     * Converts dictionary text to object
     *
     * @param {string} dictionaryText Dictionary text
     * @returns {{}} 
     */
    #dictionaryToObject(dictionaryText) {
        const entries = dictionaryText.split(/\n\s*\n/); // Split by blank lines
        const dictionary = [];
        for (let entry of entries) {
            const lines = entry.trim().split(/\r?\n/);
            if (lines.length < 8) continue; // Skip incomplete entries
            const [
                word,
                definition,
                synonymsLine,
                antonymsLine,
                examplesLine,
                partOfSpeech,
                pronunciation,
                etymology
            ] = lines;

            // Process synonyms and antonyms: remove brackets and split
            const processList = (line) => {
                return line
                    .replace(/[\[\]]/g, '') // Remove brackets
                    .split(',')
                    .map(item => item.trim())
                    .filter(item => item.length > 0);
            };
            const synonyms = processList(synonymsLine);
            const antonyms = processList(antonymsLine);
            // Process examples: remove quotes and split if multiple
            const examples = examplesLine
                .replace(/["']/g, '') // Remove quotes
                .split(';')
                .map(ex => ex.trim())
                .filter(ex => ex.length > 0);

            // Build the object
            dictionary.push({
                word: word.trim(),
                definition: definition.trim(),
                synonyms: synonyms,
                antonyms: antonyms,
                examples: examples,
                partOfSpeech: partOfSpeech.trim(),
                pronunciation: pronunciation.trim(),
                etymology: etymology.trim()
            });
        }

        return dictionary;
    }
    /**
     * Import a dictionary from URL, File, or Object
     *
     * @param {URL|File|Object} dictionary Dictionary source, must end in .dict for URL/File
     * @returns {Promise<void>}
     */
    async importDictionary(dictionary) {
        if (dictionary instanceof URL || (typeof dictionary === 'string' && (dictionary.startsWith('http://') || dictionary.startsWith('https://')))) {
            // Load from URL
            return await fetch(dictionary.toString())
                .then(response => response.text())
                .then(text => {
                    const dictObj = this.#dictionaryToObject(text);
                    for (let entry of dictObj) {
                        this.addWord(entry);
                    }
                })
                .catch(err => {
                    throw new Error("Failed to load dictionary from URL: " + err.message);
                });
        } else if (dictionary instanceof File) {
            // Load from File
            const reader = new FileReader();
            reader.onload = (e) => {
                const text = e.target.result;
                const dictObj = this.#dictionaryToObject(text);
                for (let entry of dictObj) {
                    this.addWord(entry);
                }
            };
            reader.onerror = (err) => {
                throw new Error("Failed to load dictionary from File: " + err.message);
            };
            reader.readAsText(dictionary);
            return new Promise((resolve, reject) => {
                reader.onloadend = () => resolve();
                reader.onerror = () => reject();
            });
        } else if (typeof dictionary === 'object') {
            // Load from Object
            for (let entry of dictionary) {
                this.addWord(entry);
            }
            return Promise.resolve();
        } else throw new Error("Invalid dictionary source");
    }
    /**
     * Returns the dictionary as a text file
     * @returns {string} Dictionary file text
     */
    exportDictionary() {
        const lines = [];
        for (let entry of this.dictionary) {
            const line = [
                entry.word,
                entry.definition || '',
                entry.synonyms ? entry.synonyms.join(',') : '',
                entry.antonyms ? entry.antonyms.join(',') : '',
                entry.examples ? (Array.isArray(entry.examples) ? entry.examples.join('|') : entry.examples) : '',
                entry.partOfSpeech || '',
                entry.pronunciation || '',
                entry.etymology || ''
            ].join(';');
            lines.push(line);
        }
        return lines.join('\n');
    }
    /**
     * Checks if a word exists in Wikipedia to verify spelling.
     * Uses XMLHttpRequest and a callback instead of async/await.
     * @param {string} word - The word to check.
     * @returns {string} Return the suggested
     */
    spellCheckFromURL(word) {
        const apiUrl = `https://${this.language.substring(0,2)}.wikipedia.org/w/api.php?action=query&list=search&format=json&origin=*&srsearch=${encodeURIComponent(word)}&srlimit=1`;
        const xhr = new XMLHttpRequest();
        xhr.open('GET', apiUrl, false); // false for synchronous
        try {
            xhr.send();
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                if (data.query && data.query.search && data.query.search.length > 0) {
                    const searchResults = data.query.search;
                    const foundTitle = searchResults[0].title.toLowerCase();
                    if (foundTitle === word.toLowerCase()) return word; // no correction needed
                    else return searchResults[0].title; // suggested correction
                    
                } else return ''; // no results found
                
            } else {
                console.error("HTTP error:", xhr.status);
                return '';
            }
        } catch (e) {
            console.error("Network or parsing error:", e);
            return '';
        }
    }
}