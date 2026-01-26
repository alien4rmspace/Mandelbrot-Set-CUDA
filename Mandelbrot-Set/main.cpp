#include "complex_plane.h"
#include "kernel_api.h"

#include <SFML/Graphics.hpp>
#include <optional>

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

int main()
{
    int N = 1<<20;
    float* x, * y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    launchAdd(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = std::fmax(maxError, std::fabs(y[i] - 3.0f));
    }
    std::cout << "Max Error: " << maxError << "\n";

    cudaFree(x);
    cudaFree(y);

    // ---------------------------------------------
    sf::RenderWindow window(sf::VideoMode({ 200, 200 }), "SFML works!");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    while (window.isOpen())
    {
        while (const std::optional<sf::Event> event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear();
        window.draw(shape);
        window.display();
    }

    return 0;
}
