#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "utils/Camera.h"
#include <iostream>


class CameraController {
public:
    CameraController(Trace::Camera& camera, GLFWwindow* window);
    void update(float deltaTime);

    Trace::Camera& m_camera;            // Reference to the camera to control
    GLFWwindow* m_window;        // The window to receive input from

    float2 m_arcDistTravelled;
    float m_camDist;
    float2 m_cameraPos;
    float2 m_mousePosT0;
    float2 m_mousePosTf;

    float m_yaw;// Initial yaw, can be adjusted based on the starting orientation
    float m_pitch;
    float m_sensitivity;

    float m_camDistT0;
    float m_camDistTf;

    float m_camDistX;
    float m_camDistY;

    bool m_activateMouseControl;
    bool m_activateZoomControl;

    bool hasPanned; // Flag to indicate whether panning occurred
    bool hasZoomed; // Flag to indicate whether zooming occurred

    void handleMouseControl(float deltaTime);
    void handleZoomControl(float deltaTime);

    void updateCamDist() {
        
        float3 r_hat = normalize(m_camera.getPosition() - m_camera.getTarget());

        m_camDist = m_camera.computeDistanceToTarget();
        m_camDistY = acosf(r_hat.y);
        m_camDistX = acosf(r_hat.x / sinf(m_camDistY));

    }

};

void updateMousePosition(float2& mousePos, GLFWwindow* m_window) {

    double xpos, ypos;
    glfwGetCursorPos(m_window, &xpos, &ypos);
    mousePos.x = xpos;
    mousePos.y = ypos;
}

CameraController::CameraController(Trace::Camera& camera, GLFWwindow* window)
    : m_camera(camera),
    m_window(window),
    m_camDist(0.0f),
    m_yaw(-90.0f), // Initial yaw, can be adjusted based on the starting orientation
    m_pitch(0.0f),
    m_sensitivity(0.1f),
    m_cameraPos(make_float2(0.0f, 0.0f)),
    m_mousePosT0(make_float2(0.0f, 0.0f)),
    m_mousePosTf(make_float2(0.0f, 0.0f)),
    m_camDistT0(0.0f),
    m_camDistTf(0.0f),
    m_activateMouseControl(false),
    m_activateZoomControl(false),
    hasPanned(false),
    hasZoomed(false) {

    updateCamDist();
}

void CameraController::update(float deltaTime) {
    handleMouseControl(deltaTime);
    handleZoomControl(deltaTime);
}

void CameraController::handleMouseControl(float deltaTime) {
    bool mousePressed = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    if (!m_activateMouseControl) {
        if (mousePressed) {

            m_activateMouseControl = true;
            updateMousePosition(m_mousePosT0, m_window);
        }
    }
    else {
        if (mousePressed) {

            hasPanned = true; // Set the flag to true if panning occurred

            updateMousePosition(m_mousePosTf, m_window);

            float2 delta = m_mousePosTf - m_mousePosT0;

            // Distance travelled along the circle surface at the camera's current distance
            m_arcDistTravelled += delta * 0.1f;

            // Convert the arc distance travelled to angles
            float distToTarget = m_camera.computeDistanceToTarget();
            float2 deltaAngles = m_arcDistTravelled / distToTarget;

            // Update camera's local x and y positions using polar coordinates
            updateCamDist();
            m_camDistX += deltaAngles.x;
            m_camDistY += deltaAngles.y;

            float3 origin = distToTarget * normalize(make_float3(
                cosf(m_camDistX) * sinf(m_camDistY),
                cosf(m_camDistY),
                sinf(m_camDistX) * sinf(m_camDistY)
            ));

            m_camera.setPosition(origin);
            m_camera.updateRotationMatrix();

            m_mousePosT0 = m_mousePosTf;

            m_arcDistTravelled = make_float2(0.0f, 0.0f);
        }
        else {
            m_activateMouseControl = false;
        }
    }
}

void CameraController::handleZoomControl(float deltaTime) {


    bool zoomCamera = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (!m_activateZoomControl) {
        if (zoomCamera) {
            m_activateZoomControl = true;

            double xpos, ypos;
            glfwGetCursorPos(m_window, &xpos, &ypos);
            m_camDistT0 = ypos;
        }
    }
    else {
        if (zoomCamera) {
            hasZoomed = true; // Set the flag to true if zooming occurred

            double xpos, ypos;
            glfwGetCursorPos(m_window, &xpos, &ypos);
            m_camDistTf = ypos;

            float delta = m_camDistTf - m_camDistT0;


            updateCamDist();
            m_camDist += delta * 0.01f;

            float3 cameraPosition = m_camera.getPosition();

            float3 origin = m_camDist * normalize(m_camera.getPosition() - m_camera.getTarget());

            m_camera.setPosition(origin);
            m_camera.updateRotationMatrix();
            //m_camera.update(origin, target);

            m_camDistT0 = m_camDistTf;
        }
        else {
            m_activateZoomControl = false;
        }
    }
}

