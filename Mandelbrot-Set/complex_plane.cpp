#include "complex_plane.h"

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>


ComplexPlane::ComplexPlane(unsigned short pixelWidth, unsigned short pixelHeight) :
	m_pixelWidth(pixelWidth),
	m_pixelHeight(pixelHeight),
	m_zoomCount(0),
	m_aspectRatio(static_cast<float>(pixelHeight) / static_cast<float>(pixelWidth)),	// Height / Width for aspect ratio
	m_planeCenter({ 0.f, 0.f }),
	m_planeSize({ BASE_WIDTH, BASE_HEIGHT * m_aspectRatio }),
	m_state(State::CALCULATING),
	m_vArray(sf::PrimitiveType::Points, static_cast<std::size_t>(m_pixelWidth) * static_cast<std::size_t>(m_pixelHeight))
{
	m_hostIters.resize((size_t)m_pixelWidth * (size_t)m_pixelHeight);

	cudaMalloc(&d_iters, m_hostIters.size() * sizeof(unsigned short));
}

ComplexPlane::~ComplexPlane() {
	if (d_iters) {
		cudaFree(d_iters);
	}
}

void ComplexPlane::draw(sf::RenderTarget& target, sf::RenderStates states) const {
	target.draw(m_vArray);
}

void ComplexPlane::updateRender() {
	// State is calculating.
	if (m_state == CALCULATING) {
		updateRenderCuda();

		//sf::Clock clock;

		//int incrementBy = 4;

		//for (std::size_t y = 0; y < m_pixelHeight; y++) {
		//	for (std::size_t x = 0; x < m_pixelWidth; x++) {
		//		m_vArray[x + y * m_pixelWidth].position = sf::Vector2f( static_cast<float>(x), static_cast<float>(y) );
		//		sf::Vector2i screenCoord = sf::Vector2i( x, y );
		//		sf::Vector2f mapPixel = mapPixelToCoords(screenCoord);

		//		std::uint8_t r, g, b = 0;
		//		iterationsToRGB(countIterations(mapPixel), r, g, b);
		//		m_vArray[x + y * m_pixelWidth].color = { r, g, b };
		//	}
		//}
	}
}

void ComplexPlane::updateRenderCuda() {
	m_params.width = (int)m_pixelWidth;
	m_params.height = (int)m_pixelHeight;
	m_params.centerX = m_planeCenter.x;
	m_params.centerY = m_planeCenter.y;
	m_params.sizeX = m_planeSize.x;
	m_params.sizeY = m_planeSize.y;
	m_params.maxIter = 64 + m_zoomCount * 32;

	launchMandelbrotIters(d_iters, &m_params);
	cudaDeviceSynchronize();

	cudaMemcpy(
		m_hostIters.data(),
		d_iters,
		m_hostIters.size() * sizeof(unsigned short),
		cudaMemcpyDeviceToHost
	);

	for (std::size_t y = 0; y < m_pixelHeight; y++) {
		for (std::size_t x = 0; x < m_pixelWidth; x++) {
			std::size_t idx = x + y * m_pixelWidth;

			m_vArray[idx].position = sf::Vector2f(static_cast<float>(x), static_cast<float>(y));

			std::uint8_t r = 0, g = 0, b = 0;
			iterationsToRGB(m_hostIters[idx], (size_t)m_params.maxIter, r, g, b);

			m_vArray[idx].color = sf::Color(r, g, b);
		}
	}
}

void ComplexPlane::zoomIn() {
	m_zoomCount++;
	m_zoomCount = std::min(m_zoomCount, 100);

	double newX = BASE_WIDTH * (pow(BASE_ZOOM, m_zoomCount));
	double newY = BASE_HEIGHT * m_aspectRatio * (pow(BASE_ZOOM, m_zoomCount));

	m_planeSize = { newX, newY };
	m_state = CALCULATING;
}

void ComplexPlane::zoomOut() {
	m_zoomCount--;
	m_zoomCount = std::min(m_zoomCount, 100);

	double newX = BASE_WIDTH * (pow(BASE_ZOOM, m_zoomCount));
	double newY = BASE_HEIGHT * m_aspectRatio * (pow(BASE_ZOOM, m_zoomCount));

	m_planeSize = { newX, newY };
	m_state = CALCULATING;
}

void ComplexPlane::setCenter(sf::Vector2i mousePixel) {
	sf::Vector2<double> complexCoord = mapPixelToCoords(mousePixel);

	m_planeCenter = complexCoord;
	m_state = CALCULATING;
}

void ComplexPlane::setMouseLocation(sf::Vector2i mousePixel) {
	sf::Vector2<double> complexCoord = mapPixelToCoords(mousePixel);

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

void ComplexPlane::iterationsToRGB(std::size_t iter,
	std::size_t maxIter,
	std::uint8_t& r,
	std::uint8_t& g,
	std::uint8_t& b)
{
	if (iter >= maxIter) { r = g = b = 0; return; }

	double t = double(iter) / double(maxIter); 

	if (t > 0.85) { 
		r = 255; g = 0;   b = 0; 
	}
	else if (t > 0.70) {
		r = 255; g = 255; b = 0;
	}
	else if (t > 0.55) {
		r = 255; g = 255; b = 255; 
	}
	else if (t > 0.40) {
		r = 0;   g = 255; b = 255; 
	}
	else if (t > 0.25) { 
		r = 0;   g = 0;   b = 255; 
	}
	else { 
		r = 124; g = 124; b = 124; 
	}
}


sf::Vector2<double> ComplexPlane::mapPixelToCoords(sf::Vector2i mousePixel) {
	sf::Vector2i displayPixel_x = { 0, m_pixelWidth };
	sf::Vector2i  displayPixel_y = { m_pixelHeight, 0 };

	double normalized_x = static_cast<double>(mousePixel.x - displayPixel_x.x) / (displayPixel_x.y - displayPixel_x.x);
	double normalized_y = static_cast<double>(mousePixel.y - displayPixel_y.x) / (displayPixel_y.y - displayPixel_y.x);

	double mapped_x = normalized_x * (this->m_planeSize.x) + (this->m_planeCenter.x - this->m_planeSize.x / 2);
	double mapped_y = normalized_y * (this->m_planeSize.y) + (this->m_planeCenter.y - this->m_planeSize.y / 2);

	return sf::Vector2<double>(mapped_x, mapped_y);
}