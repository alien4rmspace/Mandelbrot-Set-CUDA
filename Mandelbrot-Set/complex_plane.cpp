#include "complex_plane.h"

#include <nvtx3/nvToolsExt.h>


ComplexPlane::ComplexPlane(unsigned short pixelWidth, unsigned short pixelHeight) :
	m_pixelWidth(pixelWidth),
	m_pixelHeight(pixelHeight),
	m_zoomCount(0),
	m_aspectRatio(static_cast<float>(pixelHeight) / static_cast<float>(pixelWidth)),	// Height / Width for aspect ratio
	m_planeCenter({ 0.f, 0.f }),
	m_planeSize({ BASE_WIDTH, BASE_HEIGHT * m_aspectRatio }),
	m_state(State::CALCULATING),
	m_vArray(sf::PrimitiveType::Points, static_cast<std::size_t>(m_pixelWidth)* m_pixelHeight)
{
}

void ComplexPlane::draw(sf::RenderTarget& target, sf::RenderStates states) const {
	target.draw(m_vArray);
}

void ComplexPlane::updateRender() {
	nvtxRangePush("updateRender");

	// State is calculating.
	if (m_state == CALCULATING) {
		sf::Clock clock;

		int incrementBy = 4;

		for (std::size_t y = 0; y < m_pixelHeight; y++) {
			for (std::size_t x = 0; x < m_pixelWidth; x++) {
				m_vArray[x + y * m_pixelWidth].position = sf::Vector2f( static_cast<float>(x), static_cast<float>(y) );
				sf::Vector2i screenCoord = sf::Vector2i( x, y );
				sf::Vector2f mapPixel = mapPixelToCoords(screenCoord);

				std::uint8_t r, g, b = 0;
				iterationsToRGB(countIterations(mapPixel), r, g, b);
				m_vArray[x + y * m_pixelWidth].color = { r, g, b };
			}
		}
	}

	nvtxRangePop();

}

void ComplexPlane::zoomIn() {
	m_zoomCount++;

	float newX = BASE_WIDTH * (pow(BASE_ZOOM, m_zoomCount));
	float newY = BASE_HEIGHT * m_aspectRatio * (pow(BASE_ZOOM, m_zoomCount));

	m_planeSize = { newX, newY };
	m_state = CALCULATING;
}

void ComplexPlane::zoomOut() {
	m_zoomCount--;

	float newX = BASE_WIDTH * (pow(BASE_ZOOM, m_zoomCount));
	float newY = BASE_HEIGHT * m_aspectRatio * (pow(BASE_ZOOM, m_zoomCount));

	m_planeSize = { newX, newY };
	m_state = CALCULATING;
}

void ComplexPlane::setCenter(sf::Vector2i mousePixel) {
	sf::Vector2f complexCoord = mapPixelToCoords(mousePixel);

	m_planeCenter = complexCoord;
	m_state = CALCULATING;
}

void ComplexPlane::setMouseLocation(sf::Vector2i mousePixel) {
	sf::Vector2f complexCoord = mapPixelToCoords(mousePixel);

	m_mouseLocation = complexCoord;
}

void ComplexPlane::loadText(sf::Text& text) {
	std::stringstream stream;
	stream << "Mandelbrot Set" << std::endl;
	stream << "Center: (" << m_planeCenter.x << "," << m_planeCenter.y << ")" << std::endl;
	stream << std::fixed << std::setprecision(2) << "Cursor: (" << m_mouseLocation.x << "," << m_mouseLocation.y << ")" << std::endl;
	stream << "Left-click to Zoom in" << std::endl;
	stream << "Right-click to Zoom out" << std::endl;

	text.setString(stream.str());
}

std::size_t ComplexPlane::countIterations(sf::Vector2f coord) {
	size_t iterations = 0;
	const double threshold = 2.0;	// Diverge to infinity if > 2.0

	double re = coord.x;
	double im = coord.y;

	std::complex<double> c(re, im);
	std::complex<double> z(0, 0);

	while (abs(z) < threshold && iterations < MAX_ITER) {
		z = z * z + c;
		iterations++;
	}
	return iterations;
}

void ComplexPlane::iterationsToRGB(std::size_t count, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) {
	if (count >= 64) {
		r = 0;
		g = 0;
		b = 0;
	}
	else if (count >= 54) {
		r = 255;
		g = 0;
		b = 0;
	}
	else if (count >= 44) {
		r = 255;
		g = 255;
		b = 0;
	}
	else if (count >= 34) {
		r = 255;
		g = 255;
		b = 255;
	} 
	else if (count >= 24) {
		r = 0;
		g = 255;
		b = 255;
	}
	else if (count >= 14) {
		r = 0;
		g = 0;
		b = 255;
	}
	else if (count >= 0) {
		r = 124;
		g = 124;
		b = 124;
	}
}

sf::Vector2f ComplexPlane::mapPixelToCoords(sf::Vector2i mousePixel) {
	sf::Vector2i displayPixel_x = { 0, m_pixelWidth };
	sf::Vector2i  displayPixel_y = { m_pixelHeight, 0 };

	float normalized_x = static_cast<float>(mousePixel.x - displayPixel_x.x) / (displayPixel_x.y - displayPixel_x.x);
	float normalized_y = static_cast<float>(mousePixel.y - displayPixel_y.x) / (displayPixel_y.y - displayPixel_y.x);

	float mapped_x = normalized_x * (this->m_planeSize.x) + (this->m_planeCenter.x - this->m_planeSize.x / 2);
	float mapped_y = normalized_y * (this->m_planeSize.y) + (this->m_planeCenter.y - this->m_planeSize.y / 2);

	return sf::Vector2f(mapped_x, mapped_y);
}