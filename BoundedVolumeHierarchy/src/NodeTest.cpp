#include <iostream>
#include <algorithm>   // For std::sort
#include <random>   
#include <sutil\vec_math.h>

#include <numeric>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>
#include <chrono>

#define PLOT_SIM 1
#define PRINT_SIM 0
#define INPUT_FORMAT 0

#include "bvh_util.h"
#include "Particle.h"
#include "AABB.h"
#include "Node.h"
#include "BVH.h"

// Define floatN. it can either be float2 or float3
#if INPUT_FORMAT

ACCEL_INPUT_FORMAT inputFormat = ACCEL_INPUT_FORMAT::FLOAT2;
using floatN = float2;
using intN = int2;

#elif INPUT_FORMAT == 0

ACCEL_INPUT_FORMAT inputFormat = ACCEL_INPUT_FORMAT::FLOAT3;
using floatN = float3;
using intN = int3;

#endif // INPUT_FORMAT

auto startTimer() {
	return std::chrono::high_resolution_clock::now();
}

std::chrono::duration<double> endTimer(std::chrono::high_resolution_clock::time_point start) {
	auto end = std::chrono::high_resolution_clock::now();
	return end - start;
}

// Initialize vertices with random values between 0 and width/height
template <typename T>
void vertexInit(std::vector<T>& vertices, const int& width, const int& height);

template <typename T>
void generateWireFrame(BVH<T>* bvh, std::vector<float2>& wireframeVertices, std::vector<float>& transparency);

template <typename T>
void recursiveNodeSearchSetTransparency(BVH<T>* bvh, const int NodeIdx, std::vector<float>& transparency);

template <typename T>
void dynamicTransparencyUpdateOfDepth(BVH<T>* bvh, std::vector<float>& transparency, const int viewNode);

void updateVerticesOfBox(int boxIdx, sf::VertexArray& sfWireframe, const std::vector<float2>& wireframeVertices);
void updateTransparencyOfBox(int boxIdx, sf::VertexArray& sfWireframe, const float transparency);






int main() {
	const int width = 800;
	const int height = 800;

	const int nP = 10;


    // vertices, indices




	// Initialize vertices
	std::vector<floatN> vertices(nP);
	vertexInit(vertices, width, height);

	// Initialize primitive
	Primitive<floatN> prim = Particle<floatN>(inputFormat, vertices);


	int nAxes = 3;
#if INPUT_FORMAT
	nAxes = 2;
#endif

	auto start_sort = startTimer();
	Idx rankedIdx(nAxes, nP);
	rankedIdx.rank(prim);
	//rankedIdx.morton(prim);

	auto end_sort = endTimer(start_sort);
	std::cout << "Time to sort: " << end_sort.count() * 1000 << " ms" << std::endl;
	

	// Build tree
	auto startS = startTimer();
	//BVH bvh(prim, sortedIdx);
	BVH bvh(prim, rankedIdx);
	auto endS = endTimer(startS);
	std::cout << "Time to build tree: " << endS.count() * 1000 << " ms" << std::endl;

	// Compute bvh depth statistics
	int nodeIdx = 0;
	bvh.computeDepth(prim, nodeIdx);

	// Construct bounding boxes by traversing tree structure
	nodeIdx = 0;
	bvh.initBoundingBoxes(prim);
	//bvh.computeBoundingBoxes_BVH(prim, nodeIdx);

	//nodeIdx = 0;
	//bvh.constructBoundingBoxes(prim, nodeIdx);


#if PLOT_SIM

	sf::RenderWindow window(sf::VideoMode(width, height), "");
	sf::Vector2i topLeft(0, 0);
	window.setPosition(topLeft);
	window.clear(sf::Color::Black);

	// Wireframe properties
	sf::Color wireColor = sf::Color::Cyan;
	wireColor.a = 1;
	//size_t nWireframeVertices = 8 * (bvh.size - nP);

	// Actually, this duplicates the vertices. The real is 4 * (bvh.size - nP)
	size_t nWireframeVertices = 4 * (bvh.size - nP);

	// Generate wireframe
	std::vector<float> transparency(bvh.size, 255.0f);
	std::vector<float2> wireframeVertices(nWireframeVertices);
	generateWireFrame(&bvh, wireframeVertices, transparency);

	// Allocate memory for the wireframe
	int nBoxes = bvh.size - nP;
	int nWireframe = 8 * nBoxes;
	sf::VertexArray sfWireframe(sf::PrimitiveType::Lines, nWireframe);
	for (int i = 0; i < nWireframe; i++) {
		sfWireframe[i].color = wireColor;
	}

	for( size_t i = 0; i < nBoxes; i++ ) {

		// Update box position
		updateVerticesOfBox(i, sfWireframe, wireframeVertices);

		// Update box transparency
		updateTransparencyOfBox(i, sfWireframe, transparency[i]);

	}
	

	// Draw circles as particles so we can use 1 draw call
	sf::VertexArray sfParticles(sf::PrimitiveType::Points);
	for( int i = 0; i < nP; i++ ) {
		sfParticles.append(sf::Vertex(sf::Vector2f(prim.vertex[i].x, prim.vertex[i].y), sf::Color::White));
	}

	// Node to display
	int currentNode = 1;
	std::vector<int> previousNodes;
	previousNodes.push_back(currentNode);

	// Depth to display
	// 1 -> display all the way to the leaves
	// currentDepth == nodeDepth -> display only the current node
	int currentDepth = 1;

	// Keep window open
	while( window.isOpen() ) {
		sf::Event event;
		while( window.pollEvent(event) ) {
			// Close window: exit
			if( event.type == sf::Event::Closed ) {
				window.close();
			}

			// Check for key press events
			if( event.type == sf::Event::KeyPressed ) {

				bool leftPressed = event.key.code == sf::Keyboard::Left;
				bool rightPressed = event.key.code == sf::Keyboard::Right;
				bool downPressed = event.key.code == sf::Keyboard::Down;

				if( leftPressed || rightPressed ) {

					if (previousNodes.size() == 0) {
						previousNodes.push_back(currentNode);
					}

					if (currentNode == previousNodes[previousNodes.size() - 1]) {
						if (previousNodes.size() > 1) {
							previousNodes.pop_back();
						}
						//previousNodes.pop_back();
					} else {


						previousNodes.push_back(currentNode);  // Store the current node before updating
					}
				}

				if( downPressed ) {
					if( !previousNodes.empty() ) { // If we have previously visited nodes

						if (previousNodes.size() > 1) {
							currentNode = previousNodes[previousNodes.size() - 2];
							previousNodes.pop_back();

						} else if (previousNodes.size() == 1) {
							currentNode = previousNodes[previousNodes.size() - 1];
							previousNodes.pop_back();
						} else {
							currentNode = 1;
						}



						//currentNode = previousNodes.back(); // Go back to the last visited node
						//previousNodes.pop_back(); // Remove the last visited node from our history
					} else { // If we're already at the root node
						currentNode =1;  // Reset the nodes
						previousNodes.push_back(currentNode); // Store the root node
						//previousNodes.push_back(currentNode); // Store the root node
					}
				}

				if( leftPressed ) {
					bool isLeaf = bvh.node[currentNode].isLeaf;

					if( !isLeaf ) {
						currentNode = bvh.node[currentNode].branchIdx[0];
						std::cout << "Moved to left node: " << currentNode << std::endl;
					}
				}

				if( rightPressed ) {
					bool isLeaf = bvh.node[currentNode].isLeaf;

					if( !isLeaf ) {
						currentNode = bvh.node[currentNode].branchIdx[1];
						std::cout << "Moved to right node: " << currentNode << std::endl;
					}
				}

				// Update transparency vector
				dynamicTransparencyUpdateOfDepth(&bvh, transparency, currentNode);

				// Now we have the correct transparency vector
				// Update the transparency of the box
				for( size_t i = 0; i < nBoxes; i++ ) {
					updateTransparencyOfBox(i, sfWireframe, transparency[i]);
				}

				// Good for now
			}

		}

		window.clear();

		// Draw wireframe and particles
		window.draw(sfWireframe);
		window.draw(sfParticles);

		window.display();

	} // End of while loop

#endif // PLOT_SIM

	return 0;
}

template <typename T>
void vertexInit(std::vector<T>& vertices, const int& width, const int& height) {

	for (int i = 0; i < vertices.size(); i++) {
		// Generate non-repeating random integers between 0 and (xMax, yMax)
		//inc[i] = (float)i;

		vertices[i].x = (float)(rand() % width);
		vertices[i].y = (float)(rand() % height);
		if constexpr (std::is_same<T, float3>::value) {
			vertices[i].z = (float)(rand() % height);
		}
	}

}

template <typename T>
void generateWireFrame(BVH<T>* bvh, std::vector<float2>& wireframeVertices, std::vector<float>& transparency) {

	size_t nthInernalNode = -1;
	for (int i = 0; i < bvh->size; i++) {

		// Get node ptr
		Node<T>* node = &bvh->node[i];
		AABB<T>* box = &bvh->bbox[i];
		if (node->isLeaf) { continue; }

		// Get box
		
		//AABB<floatN>* box = node->box;
		floatN min = box->min;
		floatN max = box->max;

		// Set vertices, indices are nthInternalNode * 4 + 0, 1, 2, 3
		nthInernalNode++;
		wireframeVertices[(nthInernalNode) * 4 + 0] = make_float2(min.x, min.y);
		wireframeVertices[(nthInernalNode) * 4 + 1] = make_float2(max.x, min.y);
		wireframeVertices[(nthInernalNode) * 4 + 2] = make_float2(max.x, max.y);
		wireframeVertices[(nthInernalNode) * 4 + 3] = make_float2(min.x, max.y);

		// Set transparency to be proportional to the depth of the node
		int nodeDepth = node->depth;
		transparency[nthInernalNode] = 0.5f * ((float)nodeDepth / (float)bvh->depth);
	}
}

template <typename T>
void recursiveNodeSearchSetTransparency(BVH<T>* bvh, const int NodeIdx, std::vector<float>& transparency) {

	// Get node ptr
	Node<T>* node = &bvh->node[NodeIdx];
	if (node->isLeaf) { return; }

	// Set transparency of current node to 1
	transparency[NodeIdx] = 1.0f;

	// Call recursive function on children
	int leftChildIdx = node->branchIdx[0];
	int rightChildIdx = node->branchIdx[1];
	recursiveNodeSearchSetTransparency(bvh, leftChildIdx, transparency);
	recursiveNodeSearchSetTransparency(bvh, rightChildIdx, transparency);
}

template <typename T>
void dynamicTransparencyUpdateOfDepth(BVH<T>* bvh, std::vector<float>& transparency, const int viewNode) {
	// Recursively set transparency of all nodes below the desired node
	std::fill(transparency.begin(), transparency.end(), 0.0f);
	recursiveNodeSearchSetTransparency(bvh, viewNode, transparency);
}

void updateVerticesOfBox(int boxIdx, sf::VertexArray& sfWireframe, const std::vector<float2>& wireframeVertices) {
	std::vector<int> idx = { 0, 1, 1, 2, 2, 3, 3, 0 };
	for (int i = 0; i < 8; i++) {
		sfWireframe[boxIdx * 8 + i].position = sf::Vector2f(
			wireframeVertices[boxIdx * 4 + idx[i]].x,
			wireframeVertices[boxIdx * 4 + idx[i]].y);
	}
}

void updateTransparencyOfBox(int boxIdx, sf::VertexArray& sfWireframe, const float transparency) {
	for (int i = 0; i < 8; i++) {
		sfWireframe[boxIdx * 8 + i].color.a = transparency * 255;
	}
}


//void generateWireFrame(BVH* bvh, std::vector<float2>& wireframeVertices) {
//
//	// This function generates the wireframe vertices for the BVH
//	// It is currently being use to visualize the BVH in SFML
//	// In order to use openGL, the vertices need to be stored in a VBO
//	// The indices are stored in a VAO
//	// Sample code for openGL implementation written in comments below
//	// std::vector<float> wireframeVertices;
//	// std::vector<int> wireframeIndices;
//	// glGenVertexArrays(1, &vao);
//	// glBindVertexArray(vao);
//	// glGenBuffers(1, &vbo);
//	// glBindBuffer(GL_ARRAY_BUFFER, vbo);
//	// glBufferData(GL_ARRAY_BUFFER, wireframeVertices.size() * sizeof(float), wireframeVertices.data(), GL_STATIC_DRAW);
//	// glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
//	// glEnableVertexAttribArray(0);
//	// glGenBuffers(1, &ebo);
//	// glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
//	// glBufferData(GL_ELEMENT_ARRAY_BUFFER, wireframeIndices.size() * sizeof(int), wireframeIndices.data(), GL_STATIC_DRAW);
//	// glBindVertexArray(0);
//	//
//	// glBindVertexArray(vao);
//	// glDrawElements(GL_LINES, wireframeIndices.size(), GL_UNSIGNED_INT, 0);
//	// glBindVertexArray(0);
//	// 
//	// glDeleteVertexArrays(1, &vao);
//	// glDeleteBuffers(1, &vbo);
//	// glDeleteBuffers(1, &ebo);
//	//
//	// What isn't shown here is the shader code, which is pretty simple
//	// The vertex shader is just a passthrough
//	// The fragment shader is just a color
//	// 