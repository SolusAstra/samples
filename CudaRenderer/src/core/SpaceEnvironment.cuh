#pragma once
#include "env/Environment.cuh"

#include "core/celestial/Background.cuh"
#include "core/celestial/Earth.cuh"
#include "core/celestial/Sun.cuh"

#include "core/frame/ReferenceFrame.h"


class Space : public Trace::Environment
{

public:

    Transform transform;

    Background* galacticBackground;
    Earth* earth;
    Sun* sun;


};