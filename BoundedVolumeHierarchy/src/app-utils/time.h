#pragma once
#include <chrono>

auto getCurrentTime() {
    return std::chrono::high_resolution_clock::now();
}

std::chrono::duration<double> endTimer(std::chrono::high_resolution_clock::time_point start) {
    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

class Timer {

public:

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    Timer() {}

    void start() {
        startTime = getCurrentTime();
    }

    void reportDuration(char* label) {

        auto duration = endTimer(startTime);
        std::cout << label << duration.count() * 1000 << " ms" << std::endl;
    }
};