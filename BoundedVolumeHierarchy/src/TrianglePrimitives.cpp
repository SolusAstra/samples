#include <iostream>
#include <algorithm>   // For std::sort
#include <random>   
#include <sutil\vec_math.h>

#include <numeric>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>

#define PLOT_SIM 1
#define PRINT_SIM 0


#include "bvh_util.h"
#include "env/Particle.h"
#include "env/Triangle.h"
#include "BVH.h"

#include "app-utils/plot.h"
#include "app-utils/time.h"
#include "app-utils/bvhPlot.h"

#include "scene/premade.h"

#include "Ray.h"
#include "scene/Camera.h"
#include <AccelerationStructure.h>
#include <unordered_set>


enum SCENE_TYPE {
    RANDOM = 0,
    BUNNY = 1,
    EXEMPLAR = 2
};

// Function to update the line vertices based on the triangle indices
void updateLineVertices(sf::Vertex* line, const std::vector<float3>& vertices, 
    const std::vector<int3>& indices, int k, const sf::Color& color, Trace::Camera camera) {
    
    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;

    float3 A = vertices[indices[k].x];
    float3 B = vertices[indices[k].y];
    float3 C = vertices[indices[k].z];
    float2 vrt_px_0 = camera.worldToPixel(A, (float) WIDTH, (float) HEIGHT);
    float2 vrt_px_1 = camera.worldToPixel(B, (float) WIDTH, (float) HEIGHT);
    float2 vrt_px_2 = camera.worldToPixel(C, (float) WIDTH, (float) HEIGHT);

    // Update positions
    line[0].position = sf::Vector2f(vrt_px_0.x, vrt_px_0.y);
    line[1].position = sf::Vector2f(vrt_px_1.x, vrt_px_1.y);
    line[2].position = sf::Vector2f(vrt_px_1.x, vrt_px_1.y);
    line[3].position = sf::Vector2f(vrt_px_2.x, vrt_px_2.y);
    line[4].position = sf::Vector2f(vrt_px_2.x, vrt_px_2.y);
    line[5].position = sf::Vector2f(vrt_px_0.x, vrt_px_0.y);

    // Update colors
    for (int i = 0; i < 6; ++i) {
        line[i].color = color;
    }
}

// Function to update the line vertices based on the triangle indices
void updateNormal(sf::Vertex* line, const std::vector<float3>& vertices,
    const std::vector<int3>& indices, int k, const sf::Color& color, Trace::Camera camera) {

    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;

    float3 A = vertices[indices[k].x];
    float3 B = vertices[indices[k].y];
    float3 C = vertices[indices[k].z];

    // Flipping to negative corrects the orientation??
    float3 normal = -computeNormal(vertices, indices, k);
    float3 target = computeBarycenter(vertices, indices, k);

    float3 vertA = target;
    float3 vertB = normal + target;

    float2 vrt_px_0 = camera.worldToPixel(vertA, (float) WIDTH, (float) HEIGHT);
    float2 vrt_px_1 = camera.worldToPixel(vertB, (float) WIDTH, (float) HEIGHT);

    // Update positions
    line[0].position = sf::Vector2f(vrt_px_0.x, vrt_px_0.y);
    line[1].position = sf::Vector2f(vrt_px_1.x, vrt_px_1.y);

    // Update colors
    for (int i = 0; i < 2; ++i) {
        line[i].color = color;
    }
}

int main() {

    float aspectRatio = 16.0f / 9.0f;
    int WIDTH = 1440;
    int HEIGHT = (float) WIDTH / aspectRatio;

    int numVertices;
    int numPolygons;
    std::vector<float3> vertices(0);
    std::vector<int3> indices(0);

    Timer timer;
    Environment env;
    Trace::Camera cam;
    Primitive* primitive = nullptr;
    SCENE_TYPE scene = SCENE_TYPE::BUNNY;
    //scene = SCENE_TYPE::RANDOM;

    // Build scene
    switch (scene) {
        case (SCENE_TYPE::EXEMPLAR):
        {

            exemplar(vertices, indices);

            primitive = new Triangle(vertices, indices);

            numVertices = primitive->vertex.size();
            numPolygons = primitive->face.size();


            //TriangleEntity tEntity = trianglePrimitive.getEntity(5);
            float3 camPosition = make_float3(0.5f, 0.5f, 3.0f);
            float vFOV = 150.0f;
            cam = Trace::Camera(camPosition, camPosition, vFOV, 1.0f);
            break;
        }
        case (SCENE_TYPE::BUNNY):
        {
            stanfordBunny(vertices, indices);

            //Triangle trianglePrimitive = Triangle(vertices, indices);
            primitive = new Triangle(vertices, indices);

            numVertices = primitive->vertex.size();
            numPolygons = primitive->face.size();


            float3 camPosition = make_float3(2.0f, 3.0f, -1.0f);
            float3 camTarget = make_float3(0.0f, 0.0f, 0.0f);
            float vFOV = 40.0f;
            cam = Trace::Camera(camPosition, camTarget, vFOV, aspectRatio);
            break;
        }
        case (SCENE_TYPE::RANDOM):
        {
            numVertices = 100;
            numPolygons = numVertices;
            randomizePoints(vertices, numVertices, WIDTH, HEIGHT);

            primitive = new Particle(vertices);

            float3 camPosition = make_float3(0.0f, 2.0f, 1.0f);
            float3 camTarget = make_float3(0.0f, 0.0f, 0.0f);
            float vFOV = 40.0f;
            cam = Trace::Camera(camPosition, camTarget, vFOV, aspectRatio);
            //trianglePrimitive = Particle<float3>::Positions(vertices);
        }
    }


    // Problem Definition
    std::cout << "# Vertices: " << numVertices << std::endl;
    std::cout << "# Polygons: " << numPolygons << std::endl;


    // Bounded Volume Heirarchy
    timer.start();
    AccelStruct::BVH bvh(primitive);

    char msgName[] = "Time to build tree: ";
    timer.reportDuration(msgName);

    // Linear
    AccelStruct::BruteForce linearSearch(primitive);

    // Compute acceleration structure statistics
    int nodeIdx = 0;
    computeDepth(&bvh, nodeIdx);
    initBoundingBoxes(&bvh, primitive);


    dPrimitive* dprimitive = primitive->createDeviceVersion();

    // Make device version
    //AccelStruct::dBVH dbvh(&bvh);



#if PLOT_SIM

    sf::VertexArray sfPrimitives;
    sf::VertexArray sfWireframe;
    NodeState nodeState;
    nodeState.push();

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "");
    window.setPosition(sf::Vector2i(0, 0));
    window.clear(sf::Color::Black);


    sf::Vertex line[6];


    sf::Vertex normal[2];


    // Generate SFML primitives for plotting

    switch (scene) {

        case (SCENE_TYPE::EXEMPLAR):
        {
            sfPrimitives = generateSFMLPrimitives<sf::PrimitiveType::Triangles>(primitive->vertex, primitive->face, cam);
            sfWireframe = buildWireFrame(bvh, cam);
            break;
        }
        case (SCENE_TYPE::BUNNY):
        {
            sfPrimitives = generateSFMLPrimitives<sf::PrimitiveType::Triangles>(primitive->vertex, primitive->face, cam);
            sfWireframe = buildWireFrame(bvh, cam);
            break;
        }
        case (SCENE_TYPE::RANDOM):
        {
            sfPrimitives = generateSFMLPrimitives<sf::PrimitiveType::Points>(primitive->vertex, primitive->face, cam);
            sfWireframe = buildWireFrame(bvh);
            break;
        }
    }


    sf::Event events;

    // Rendering Loop
    while (window.isOpen()) {

        // Process events
        window.pollEvent(events);

        // Check for key press events
        bool leftPressed = events.key.code == sf::Mouse::Left && events.type == sf::Event::MouseButtonPressed;

        if (leftPressed) {

            // Get current mouse position
            sf::Vector2i mousePosition = sf::Mouse::getPosition();

            // Extend ray from camera to clicked location
            //float3 rayDir = cam.pixelToDirection(float(position.x), float(position.y), float(WIDTH), float(HEIGHT));
            float i = (float(WIDTH) - float(mousePosition.x)) / float(WIDTH);
            float j = (float(HEIGHT) - float(mousePosition.y)) / float(HEIGHT);

            Payload payload;
            Trace::Ray ray(cam.getPosition(), cam.getPixelPosition(i, j));

            Environment::hit(&bvh, primitive, ray, payload);

            //Environment::hit(&linearSearch, ray, payload);

            if (payload.wasHit) {

                int nodeIdx = 0;
                bool primFound = false;
                std::vector<int> path(0);
                int primIdx = payload.primitiveID;
                getPath(&bvh, nodeIdx, primFound, primIdx, path);

                std::cout << "idx: " << primIdx << std::endl;

                sf::Color color = shadePrimitives(primIdx, vertices, indices, cam, FLT_MAX);

                updateLineVertices(line, vertices, indices, primIdx, sf::Color::Red, cam);

                updateNormal(normal, vertices, indices, primIdx, sf::Color::Red, cam);

                sfPrimitives[primIdx * 3 + 0].color = color;
                sfPrimitives[primIdx * 3 + 1].color = color;
                sfPrimitives[primIdx * 3 + 2].color = color;

                switch (scene) {

                    case (SCENE_TYPE::EXEMPLAR):
                    {
                        sfWireframe = buildWireFrame(bvh, path, cam);
                        break;
                    }
                    case (SCENE_TYPE::BUNNY):
                    {
                        sfWireframe = buildWireFrame(bvh, path, cam);
                        break;
                    }
                    case (SCENE_TYPE::RANDOM):
                    {
                        sfWireframe = buildWireFrame(bvh, path);
                        break;
                    }
                }

            } // if payload.hit

        } // If left pressed



        // Clear last render
        window.clear();

        // Draw wireframe and primitives
        window.draw(sfPrimitives);
        window.draw(sfWireframe);
        window.draw(line, 6, sf::Lines);
        window.draw(normal, 2, sf::Lines);
        window.display();

    } // while(window.isOpen())

#endif // PLOT_SIM


    delete primitive;
    return 0;
}
