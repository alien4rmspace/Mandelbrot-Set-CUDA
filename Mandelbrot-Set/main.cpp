#include "complex_plane.h"
#include "kernel_api.h"

#include <SFML/Graphics.hpp>
#include <optional>

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>


void handleInput(sf::RenderWindow& window, ComplexPlane& complexPlane) {
    while (const auto event = window.pollEvent())
    {
        // Close
        if (event->is<sf::Event::Closed>())
        {
            window.close();
            continue;
        }

        // Key pressed
        if (const auto* key = event->getIf<sf::Event::KeyPressed>())
        {
            if (key->code == sf::Keyboard::Key::Escape)
            {
                window.close();
            }
            else if (key->code == sf::Keyboard::Key::Space)
            {
                std::cout << "loaded" << std::endl;
            }
            continue;
        }

        // Mouse button pressed
        if (const auto* mouse = event->getIf<sf::Event::MouseButtonPressed>())
        {
            std::cout << ((mouse->button == sf::Mouse::Button::Left) ? "Left" : "Right")
                << " mouse button was pressed" << std::endl;
            std::cout << "mouse x: " << mouse->position.x << std::endl;
            std::cout << "mouse y: " << mouse->position.y << std::endl;

            if (mouse->button == sf::Mouse::Button::Left)
            {
                complexPlane.zoomIn();
                complexPlane.setCenter({ static_cast<int>(mouse->position.x),
                                         static_cast<int>(mouse->position.y) });
            }
            else if (mouse->button == sf::Mouse::Button::Right)
            {
                complexPlane.zoomOut();
            }
            continue;
        }

        // Mouse moved
        if (const auto* moved = event->getIf<sf::Event::MouseMoved>())
        {
            complexPlane.setMouseLocation(
                { static_cast<int>(moved->position.x),
                  static_cast<int>(moved->position.y) }
            );
            continue;
        }
    }
}

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
	sf::RenderWindow window;
	const sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
	unsigned short windowWidth = desktop.size.x / 2;
	unsigned short windowHeight = desktop.size.y / 2;
	window.create(
		sf::VideoMode({ windowWidth, windowHeight }),
		"Mandelbrot",
		sf::Style::Default
	);
	sf::Font font;
	font.openFromFile("ARIAL.ttf");
	sf::Text text(font);
	text.setCharacterSize(20);

	ComplexPlane complexPlane(windowWidth, windowHeight);

	while (window.isOpen()) {
		handleInput(window, complexPlane);
		complexPlane.updateRender();
		complexPlane.loadText(text);

		// Drawing Stage
		window.clear();
		window.draw(complexPlane);
		window.draw(text);

		window.display();
	}

    return 0;
}
