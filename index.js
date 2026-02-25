import tf from "@tensorflow/tfjs";
import ModelService from "./service/ModelService.js";

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

const model = await ModelService.trainModel(inputXs, outputYs);

const person = {
  nome: "Ryan",
  idade: 28,
  cor: "verde",
  localizacao: "Curitiba",
};

const minAge = 25;
const maxAge = 40;
const getPersonAge = (age) => {
  const result = (age - minAge) / (maxAge - minAge);

  return result;
};

const normalizedPersonTensor = [
  [
    getPersonAge(person.idade),
    person.cor == "azul" ? 1 : 0,
    person.cor == "vermelho" ? 1 : 0,
    person.cor == "verde" ? 1 : 0,
    person.localizacao == "São Paulo" ? 1 : 0,
    person.localizacao == "Rio" ? 1 : 0,
    person.localizacao == "Curitiba" ? 1 : 0,
  ],
];

const prediction = await ModelService.predict(model, normalizedPersonTensor);

const results = prediction
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`);

console.log(results);
