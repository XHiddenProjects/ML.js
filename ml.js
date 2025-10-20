import { NLP } from "./mechine_learning.js";

const nlp = new NLP();

// Add some documents for different intents
nlp.addLanguage('en');
nlp.addDocument("Hello, how are you?", "greeting");
nlp.addDocument("Hi there!", "greeting");
nlp.addDocument("Goodbye!", "farewell");
nlp.addDocument("See you later.", "farewell");
nlp.addDocument("What's the weather like?", "weather");
nlp.addDocument("Is it sunny today?", "weather");

// Add possible answers for each intent
nlp.addAnswer("greeting", "Hello! How can I assist you?");
nlp.addAnswer("greeting", "Hi! What can I do for you?");
nlp.addAnswer("farewell", "Goodbye! Have a nice day!");
nlp.addAnswer("farewell", "See you later!");
nlp.addAnswer("weather", "The weather is sunny and warm.");
nlp.addAnswer("weather", "It's cloudy today.");

// Train the model
nlp.train();

// Process an input and get a response
const inputText = "Helloo";
const result = nlp.process(inputText);
const responseAnswers = result.answers;
const randIndex = Math.floor(Math.random()*responseAnswers.length);
console.log(responseAnswers.length > 0 ? responseAnswers[randIndex] : "Sorry, I didn't understand that.");






