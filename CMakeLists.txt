cmake_minimum_required(VERSION 4.2)
project(CircleNeuralNetwork)

set(CMAKE_CXX_STANDARD 23)

add_executable(CircleNeuralNetwork main.cpp
        neural_network.h)
