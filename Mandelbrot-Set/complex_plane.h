#pragma once
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include <iostream>
#include <complex>
#include <thread>
#include <iomanip>
#include <fstream>

const unsigned int MAX_ITER = 64;
const float BASE_WIDTH = 4.0;
const float BASE_HEIGHT = 4.0;
const float BASE_ZOOM = 0.5;

enum State { CALCULATING, DISPLAYING };

class ComplexPlane : public sf::Drawable {
public:
	ComplexPlane(unsigned short pixelWidth, unsigned short pixelHeight);
	void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
	void updateRender();
	void zoomIn();
	void zoomOut();
	void setCenter(sf::Vector2i mousePixel);
	void setMouseLocation(sf::Vector2i mousePixel);
	void loadText(sf::Text& text);

private:
	unsigned short m_pixelWidth;
	unsigned short m_pixelHeight;
	unsigned short m_zoomCount;
	float m_aspectRatio;

	State m_state;

	sf::Vector2i m_pixel_size;

	sf::Vector2f m_mouseLocation;
	sf::Vector2f m_planeCenter;
	sf::Vector2f m_planeSize;

	sf::VertexArray m_vArray;

	std::size_t countIterations(sf::Vector2f coord);
	void iterationsToRGB(std::size_t count,
						 std::uint8_t& r,
						 std::uint8_t& g,
						 std::uint8_t& b);
	sf::Vector2f mapPixelToCoords(sf::Vector2i mousePixel);
};

