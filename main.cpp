#include <iostream>
#include <iomanip>
#include <format>
#include <random>
#include <thread>
#include <stop_token>
#include <chrono>
#include <cmath>
#include "neural_network.h"

int main()
{
    neural_network<2, 16, 16, 1> circleClassifier(0.001);

    constexpr int rows = 30,
                  columns = 60;
    constexpr double centerX = columns / 2.0,
                     centerY = rows / 2.0,
                     radius = columns / 8.0;

    auto moveCursor = [](int row, int column)
    {
        std::cout << std::format("\033[{};{}H", row + 1, column + 1);
    };

    auto insideCircle = [](double x, double y)
    {
        const double distance = std::sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
        return distance < radius;
    };

    std::random_device rd;
    std::uniform_real_distribution<double> xDist(0, columns - 1),
                                           yDist(0, rows - 1),
                                           radiusDist(0, radius),
                                           thetaDist(0, 3.14159 * 2);

    std::cout << "\033[?25l";
    std::cout << "\033[2J" << std::flush;

    std::jthread thr{[&](std::stop_token st)
    {
        while (!st.stop_requested())
        {
            moveCursor(0, 0);
            for (int row = 0; row < rows; ++row)
            {
                for (int column = 0; column < columns; ++column)
                {
                    const double x = column,
                                 y = row,
                                 normalizedX = x / columns,
                                 normalizedY = y / rows;

                    const bool prediction = circleClassifier.predict({normalizedX, normalizedY})[0] > 0.5;

                    std::cout << (prediction ? 'o' : ' ');
                }

                std::cout << '\n' << std::flush;
            }

            for (int epoch = 0; epoch < 100; ++epoch)
            {
                const double x = xDist(rd),
                             y = yDist(rd),
                             normalizedX = x / columns,
                             normalizedY = y / rows;

                circleClassifier.fit({normalizedX, normalizedY}, std::vector<double>(1, insideCircle(x, y)));
            }

            for (int epoch = 0; epoch < 90; ++epoch)
            {
                const double x = centerX + radiusDist(rd) * std::cos(thetaDist(rd)),
                             y = centerY + radiusDist(rd) * std::sin(thetaDist(rd)),
                             normalizedX = x / columns,
                             normalizedY = y / rows;

                circleClassifier.fit({normalizedX, normalizedY}, std::vector<double>(1, insideCircle(x, y)));
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }};
    std::cin.get();

    std::cout << "\033[?25h";
}