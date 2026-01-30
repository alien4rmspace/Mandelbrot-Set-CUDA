#pragma once
#include "kernel_api.h"
#include "mandelbrot_params.h"

#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include <iostream>
#include <complex>
#include <thread>
#include <iomanip>
#include <fstream>
#include <algorithm>

const unsigned int MAX_ITER = 64;
const float BASE_WIDTH = 4.0;
const float BASE_HEIGHT = 4.0;
const float BASE_ZOOM = 0.5;

enum State { CALCULATING, DISPLAYING };


class ComplexPlane : public sf::Drawable {
public:
	ComplexPlane(unsigned short pixelWidth, unsigned short pixelHeight);
	~ComplexPlane();
	void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
	void updateRender();
	void updateRenderCuda();
	void zoomIn();
	void zoomOut();
	void setCenter(sf::Vector2i mousePixel);
	void setMouseLocation(sf::Vector2i mousePixel);
	void loadText(sf::Text& text);

private:
	unsigned short m_pixelWidth;
	unsigned short m_pixelHeight;
	int m_zoomCount;
	float m_aspectRatio;

	std::vector<unsigned short> m_hostIters;	// h or host for member of CPU.
	unsigned short* d_iters = nullptr;	// d for member of device.

	MandelbrotParams m_params{};

	State m_state;

	sf::Vector2i m_pixel_size;

	sf::Vector2<double> m_mouseLocation;
	sf::Vector2<double> m_planeCenter;
	sf::Vector2<double> m_planeSize;

	sf::VertexArray m_vArray;

	std::size_t countIterations(sf::Vector2<double> coord);
	void iterationsToRGB(std::size_t iter,
						 std::size_t maxIter,
						 std::uint8_t& r,
						 std::uint8_t& g,
						 std::uint8_t& b);
	sf::Vector2<double> mapPixelToCoords(sf::Vector2i mousePixel);
};

