//
// Created by cartercpp on 7/5/26.
//

#ifndef CIRCLENEURALNETWORK_NEURAL_NETWORK_H
#define CIRCLENEURALNETWORK_NEURAL_NETWORK_H

#include <stdexcept>
#include <utility>
#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <cstddef>

template <int... Is> requires (sizeof...(Is) > 0)
class neural_network
{
private:

    using WeightMatrix = std::vector<std::vector<double>>;
    using BiasVector = std::vector<double>;

public:

    explicit neural_network(double learningRate)
        : m_learningRate{learningRate}
    {
        m_weightMatrices.reserve(s_layers);
        m_biasVectors.reserve(s_layers);

        std::random_device rd;
        std::uniform_real_distribution<double> weightDist(-1, 1),
                                                biasDist(-0.5, 0.5);

        for (int layer = 1; layer < s_layers; ++layer)
        {
            const int layerSize = s_neuronsPerLayer[layer],
                      prevLayerSize = s_neuronsPerLayer[layer - 1];

            WeightMatrix weightMatrix(layerSize, std::vector<double>(prevLayerSize, 0));
            BiasVector biasVector(layerSize, 0);

            for (auto& rowRef : weightMatrix)
                for (auto& valueRef : rowRef)
                    valueRef = weightDist(rd);

            for (auto& valueRef : biasVector)
                valueRef = biasDist(rd);

            m_weightMatrices.emplace_back(std::move(weightMatrix));
            m_biasVectors.emplace_back(std::move(biasVector));
        }
    }

    std::vector<double> predict(const std::vector<double>& input) const
    {
        return Forward(input).back();
    }

    void fit(const std::vector<double>& input, const std::vector<double>& target, std::size_t epochs = 1)
    {
        for (std::size_t epoch = 0; epoch < epochs; ++epoch)
            Fit(input, target);
    }

    void fit(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs,
            std::size_t epochs)
    {
        if (inputs.size() != outputs.size())
            throw std::invalid_argument{"There must be as many inputs as outputs"};

        for (std::size_t epoch = 0; epoch < epochs; ++epoch)
            for (std::size_t i = 0; i < inputs.size(); ++i)
                Fit(inputs[i], outputs[i]);
    }

private:

    std::vector<std::vector<double>> Forward(const std::vector<double>& input) const
    {
        if (input.size() != s_neuronsPerLayer[0])
            throw std::invalid_argument{"Input size is wrong"};

        std::vector<std::vector<double>> activations;
        activations.reserve(s_layers);
        activations.push_back(input);

        for (int layer = 0; layer < s_layers - 1; ++layer)
        {
            const std::vector<double>& sourceActivations{activations[layer]};
            const WeightMatrix& sourceWeights{m_weightMatrices[layer]};
            const BiasVector& sourceBiases{m_biasVectors[layer]};
            const int sourceSize = s_neuronsPerLayer[layer];

            const int destSize = s_neuronsPerLayer[layer + 1];
            std::vector<double> destActivations(destSize, 0);

            for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
            {
                double weightedSum = sourceBiases[destNeuron];

                for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
                    weightedSum += sourceActivations[sourceNeuron] * sourceWeights[destNeuron][sourceNeuron];

                destActivations[destNeuron] = Sigmoid(weightedSum);
            }

            activations.emplace_back(std::move(destActivations));
        }

        return activations;
    }

    void Fit(const std::vector<double>& input, const std::vector<double>& target)
    {
        const int outputSize = s_neuronsPerLayer[s_layers - 1];

        if (target.size() != outputSize)
            throw std::invalid_argument{"Target size is wrong"};

        const std::vector<std::vector<double>> activations{Forward(input)};
        const std::vector<double>& prediction{activations.back()};

        std::vector<double> delta(outputSize, 0);
        for (int i = 0; i < outputSize; ++i)
            delta[i] = prediction[i] - target[i];

        for (int layer = s_layers - 2; layer >= 0; --layer)
        {
            WeightMatrix& sourceWeightsRef{m_weightMatrices[layer]};
            BiasVector& sourceBiasesRef{m_biasVectors[layer]};
            const std::vector<double>& sourceActivations{activations[layer]};

            const int sourceSize = s_neuronsPerLayer[layer],
                      destSize = s_neuronsPerLayer[layer + 1];

            std::vector<double> newDelta(sourceSize, 0);
            for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
            {
                double error = 0;

                for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
                    error += delta[destNeuron] * sourceWeightsRef[destNeuron][sourceNeuron];

                newDelta[sourceNeuron] = error * ((layer > 0) ? SigmoidDerivative(sourceActivations[sourceNeuron]): 1);
            }

            for (int destNeuron = 0; destNeuron < destSize; ++destNeuron)
            {
                for (int sourceNeuron = 0; sourceNeuron < sourceSize; ++sourceNeuron)
                    sourceWeightsRef[destNeuron][sourceNeuron]
                        -= m_learningRate * delta[destNeuron] * sourceActivations[sourceNeuron];

                sourceBiasesRef[destNeuron] -= m_learningRate * delta[destNeuron];
            }

            std::swap(delta, newDelta);
        }
    }

    static double Sigmoid(double x)
    {
        return 1 / (1 + std::exp(-x));
    }

    static double SigmoidDerivative(double x)
    {
        return x * (1 - x);
    }

    static constexpr int s_layers = static_cast<int>(sizeof...(Is));
    static constexpr std::array<int, s_layers> s_neuronsPerLayer{Is...};

    std::vector<WeightMatrix> m_weightMatrices;
    std::vector<BiasVector> m_biasVectors;
    double m_learningRate;
};

#endif //CIRCLENEURALNETWORK_NEURAL_NETWORK_H