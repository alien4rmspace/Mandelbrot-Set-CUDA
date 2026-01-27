#pragma once
// Struct needs to be separate from SFML code otherwise C++17 
// version problems from setup will arise.
struct MandelbrotParams {
	int width, height;
	double centerX, centerY;
	double sizeX, sizeY;
	int maxIter;
};